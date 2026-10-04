"""env worker:每进程串行驱动 M 个 env_fixed,决策批量提交推理服务器。

- action_mask.sum()==1 的状态本地直接执行,不走队列、不入轨迹
- 训练模式:per-game 抽模式(75% 四座当前策略全采集 / 25% 当前策略 1 座 + 池对手 3 座)
- 评测模式:固定 assignments(复式轮座)由 eval 进程使用同一驱动
- 轨迹:per 收集座位 (obs,mask,act,logp,value) + 终局 reward/30 -> GAE -> traj_q
- 计时拆分:env_step / 特征渲染 / 向听 / 队列等待,周期报 stats_q
"""
import os
import queue
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

TIMERS = {'env_step': 0.0, 'render': 0.0, 'shanten': 0.0, 'vec': 0.0, 'queue_wait': 0.0,
          'submit_put': 0.0, 'reset': 0.0, 'ctrl_ipc': 0.0, 'finish_put': 0.0}


def _install_timers():
    import feature_v1 as fv1
    orig_obs = fv1.FeatureAgentV1._obs

    def timed_obs(self):
        t0 = time.perf_counter()
        r = orig_obs(self)
        TIMERS['render'] += time.perf_counter() - t0
        return r
    fv1.FeatureAgentV1._obs = timed_obs

    import shanten_py
    for fname in ('regular_shanten_useful', 'seven_pairs_shanten_useful',
                  'thirteen_orphans_shanten_useful', 'honors_knitted_shanten_useful',
                  'knitted_straight_shanten_useful'):
        orig = getattr(shanten_py, fname)

        def timed(hand, _o=orig):
            t0 = time.perf_counter()
            r = _o(hand)
            TIMERS['shanten'] += time.perf_counter() - t0
            return r
        setattr(shanten_py, fname, timed)


class Game:
    """一局的运行时状态。assignment: seat -> (model_id, argmax, collect)。"""

    __slots__ = ('env', 'assignment', 'traj', 'pending', 'mode', 'logger', 'meta', 'cur_obs')

    def __init__(self, env, assignment, mode, logger=None, meta=None):
        self.env = env
        self.assignment = assignment
        self.mode = mode
        self.logger = logger
        self.meta = meta or {}
        self.traj = {s: [] for s, (m, am, c) in assignment.items() if c}
        self.pending = {}   # req_id -> (name, seat, obs_dict)
        self.cur_obs = None


def compute_gae(traj, final_r, gamma, lam):
    """traj: list of (obs,vec,mask,act,logp,value)。返回 (adv, ret) 数组。"""
    T = len(traj)
    values = np.array([t[5] for t in traj], np.float32)
    adv = np.zeros(T, np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        v_next = values[t + 1] if t + 1 < T else 0.0
        r = final_r if t == T - 1 else 0.0
        delta = r + gamma * v_next - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def run_env_worker(worker_id, cfg, req_q, resp_q, traj_q, ctrl, stats_q,
                   arena_name=None, slot_sem=None,
                   eval_cmd_q=None, eval_res_q=None, log_games=0):
    """训练 worker(eval_cmd_q=None)或评测 worker。"""
    import torch
    torch.set_num_threads(1)
    try:
        os.nice(5)   # 让出调度优先级给推理服务器(其饥饿会卡死全局吞吐)
    except OSError:
        pass
    _install_timers()
    from feature_v1 import FeatureAgentV1
    from feature_v2 import compute_vec_from_obs
    from env_fixed import MahjongGBEnvFixed
    from rl.game_logger import GameLogger
    from rl.common import TRAJ_SLOTS, TRAJ_SLOT_BYTES, write_traj

    shm = None
    slot_next = [0]
    if arena_name:
        from multiprocessing import shared_memory, resource_tracker
        shm = shared_memory.SharedMemory(name=arena_name)
        try:   # 防 3.12 resource_tracker 在子进程退出时误 unlink
            resource_tracker.unregister(shm._name, 'shared_memory')
        except Exception:
            pass
        my_base = worker_id * TRAJ_SLOTS * TRAJ_SLOT_BYTES

    def ship_traj(traj):
        if shm is None:
            traj_q.put(('inline', worker_id, traj, len(traj['act'])))
            return
        slot = slot_next[0]
        slot_sem.acquire()    # C 级阻塞,GIL 释放;读端消费后 release
        view = shm.buf[my_base + slot * TRAJ_SLOT_BYTES:
                       my_base + (slot + 1) * TRAJ_SLOT_BYTES]
        T = write_traj(view, traj)
        traj_q.put(('traj', worker_id, slot, T))
        slot_next[0] = (slot + 1) % TRAJ_SLOTS

    rng = random.Random(str((cfg.seed, 'worker', worker_id)))
    rid = [0]
    logged_games = []

    def new_req_id():
        rid[0] += 1
        return rid[0]

    def make_env():
        return MahjongGBEnvFixed(config={'agent_clz': FeatureAgentV1})

    def sample_assignment():
        t0 = time.perf_counter()
        snaps = list(ctrl.get('snapshots', []))
        fixed = set(ctrl.get('fixed', []))
        TIMERS['ctrl_ipc'] += time.perf_counter() - t0
        # 加权对手类别:仅纳入权重>0 且模型可用的类
        cats = []
        if cfg.w_self_snap > 0:
            cats.append(('__snap__', cfg.w_self_snap))
        if cfg.w_anchor > 0 and 'sl' in fixed:
            cats.append(('sl', cfg.w_anchor))
        if cfg.w_bighand > 0 and 'bighand' in fixed:
            cats.append(('bighand', cfg.w_bighand))
        if cfg.w_def > 0 and 'def' in fixed:
            cats.append(('def', cfg.w_def))
        if cfg.w_rl_v1b > 0 and 'rl_v1b' in fixed:
            cats.append(('rl_v1b', cfg.w_rl_v1b))
        if rng.random() < cfg.p_pure_selfplay or not cats:
            return {s: ('cur', False, True) for s in range(4)}, 'self'
        total = sum(w for _, w in cats)
        cur_seat = rng.randrange(4)
        asg = {}
        for s in range(4):
            if s == cur_seat:
                asg[s] = ('cur', False, True)
                continue
            x = rng.random() * total
            acc, mid = 0.0, cats[-1][0]
            for cid, w in cats:
                acc += w
                if x <= acc:
                    mid = cid
                    break
            if mid == '__snap__':
                mid = rng.choice(snaps) if snaps else 'cur'
            asg[s] = (mid, False, False)
        return asg, 'mixed'

    def start_game(eval_task=None):
        env = make_env()
        if eval_task is None:
            asg, mode = sample_assignment()
            wind = rng.randrange(4)
            obs = env.reset(prevalentWind=wind)
            logger = GameLogger(env, wind) if log_games and len(logged_games) < log_games else None
            g = Game(env, asg, mode, logger)
        else:
            asg, wall, wind, meta = eval_task
            obs = env.reset(prevalentWind=wind, tileWall=wall)
            g = Game(env, asg, 'eval', meta=meta)
        g.cur_obs = obs
        return g

    rulebot = [None]
    local_nn = {}        # 本地推理对手(rl_v1_B 是 v1 模型,不走 v2 infer_server)

    def submit_or_local(g):
        """对当前 obs 的每个决策:单合法本地执行,rulebot 本地推理,否则发请求。"""
        actions = {}
        for name, o in g.cur_obs.items():
            seat = int(name.split('_')[1]) - 1
            legal = np.flatnonzero(o['action_mask'])
            if len(legal) == 1:
                actions[name] = int(legal[0])
                continue
            mid, argmax, collect = g.assignment[seat]
            if mid == 'rulebot':
                if rulebot[0] is None:
                    from arena.policies import RuleBotPolicy
                    rulebot[0] = RuleBotPolicy()
                actions[name] = rulebot[0].act(o)
                continue
            if mid == 'rl_v1b':
                if local_nn.get('rl_v1b') is None:
                    from model_v1 import MahjongNetV1, InputDictAdapter
                    from arena.policies import NNPolicy
                    _net, _ = MahjongNetV1.load(cfg.rl_v1b_ckpt)
                    local_nn['rl_v1b'] = NNPolicy(InputDictAdapter(_net), temperature=1.0)
                pol = local_nn['rl_v1b']
                pol.temperature = 0.0 if argmax else 1.0   # eval 确定性 / 训练采样
                actions[name] = pol.act(o)
                continue
            r = new_req_id()
            tv = time.perf_counter()
            vec = compute_vec_from_obs(o['observation'])
            TIMERS['vec'] += time.perf_counter() - tv
            g.pending[r] = (name, seat, o, vec)
            t0 = time.perf_counter()
            req_q.put((worker_id, r, mid, o['observation'], vec, o['action_mask'], argmax))
            TIMERS['submit_put'] += time.perf_counter() - t0
        return actions

    def finish_game(g):
        rewards = [g.env.reward[i] if g.env.reward else 0 for i in range(4)]
        if g.logger:
            logged_games.append((g.logger.lines, {s: [t[:6] for t in g.traj[s]]
                                                  for s in g.traj}))
        # 番数 + 点炮归属(裁判语义,从 env_fixed 得分向量解码;eval/训练共用)
        win = int(np.argmax(rewards)) if max(rewards) > 0 else -1
        fan, dealer_in = 0, -1
        if win >= 0:
            losers = sorted(r for i, r in enumerate(rewards) if i != win)
            self_drawn = losers[0] == losers[-1]    # 自摸:三家失分相等
            fan = rewards[win] // 3 - 8 if self_drawn else rewards[win] - 24
            if not self_drawn:                       # 点和:点炮者=失分最多者
                dealer_in = int(np.argmin(rewards))
        if g.mode == 'eval':
            eval_res_q.put({'meta': g.meta, 'scores': rewards,
                            'win': win, 'fan': int(fan), 'dealer_in': dealer_in})
            return
        # α/β 从 ctrl 读(主跑红线触发时 learner 会下调 α 上调 β,跨进程生效)
        alpha = ctrl.get('reward_alpha', cfg.reward_alpha)
        beta = ctrl.get('reward_beta', cfg.reward_beta)
        fan_b = min(fan, cfg.reward_fan_cap) if cfg.reward_fan_cap > 0 else fan
        fan_d = min(fan, cfg.dealin_fan_cap) if cfg.dealin_fan_cap > 0 else fan
        fan_fb = min(fan, cfg.fan_backward_fan_cap) if cfg.fan_backward_fan_cap > 0 else fan
        t0 = time.perf_counter()
        for seat, steps in g.traj.items():
            if not steps:
                continue
            # 基础 score/scale + 单步番奖励(α,batch-2 默认关) − 番数敏感点炮罚 β·(fan/8)^p_def;
            # fan backward(进攻信用)加在 advantage 上(只动 adv,不动 ret/value 目标)
            fr = rewards[seat] / cfg.reward_scale
            if seat == win and fan >= cfg.reward_fan_thresh:
                fr += alpha * (fan_b / 8.0) ** cfg.reward_fan_pow
            if seat == dealer_in:
                fr -= beta * (fan_d / 8.0) ** cfg.dealin_pow
            adv, ret = compute_gae(steps, fr, cfg.gamma, cfg.gae_lambda)
            if seat == win and cfg.fan_backward_coef > 0 and fan >= cfg.fan_backward_min_fan:
                Tn = len(steps)
                gfan = (fan_fb / 8.0) ** 2
                kdist = np.arange(Tn - 1, -1, -1, dtype=np.float32)   # 距和牌步数:末步=0
                adv = adv + cfg.fan_backward_coef * (cfg.fan_backward_gamma ** kdist) * gfan
            ship_traj({
                'obs': np.stack([t[0] for t in steps]),
                'vec': np.stack([t[1] for t in steps]),
                'mask': np.stack([t[2] for t in steps]),
                'act': np.array([t[3] for t in steps], np.int64),
                'logp': np.array([t[4] for t in steps], np.float32),
                'adv': adv, 'ret': ret,
            })
        stats_q.put(('game', {'mode': g.mode, 'win': win, 'fan': fan,
                              'scores': rewards,
                              'hu': max(rewards) > 0,
                              'n_steps': sum(len(s) for s in g.traj.values())}))
        TIMERS['finish_put'] += time.perf_counter() - t0

    # ---------------- 主循环 ----------------
    n_envs = cfg.envs_per_worker
    eval_mode = eval_cmd_q is not None
    games = []
    eval_tasks = []
    t_report = time.time()

    while not ctrl.get('stop', False):
        if eval_mode and not games and not eval_tasks:
            try:
                cmd = eval_cmd_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if cmd[0] == 'eval':
                eval_tasks = list(cmd[1])   # [(assignment, wall, wind, meta), ...]
            continue
        while len(games) < n_envs and (not eval_mode or eval_tasks):
            t0 = time.perf_counter()
            games.append(start_game(eval_tasks.pop(0) if eval_mode else None))
            TIMERS['reset'] += time.perf_counter() - t0
            if eval_mode and not eval_tasks:
                break
        if not games:
            continue

        # 1) 提交全部待决策
        local_actions = []
        for g in games:
            acts = submit_or_local(g)
            local_actions.append(acts)
        # 2) 等响应
        n_wait = sum(len(g.pending) for g in games)
        t0 = time.perf_counter()
        got = 0
        responses = {}
        while got < n_wait:
            req_id, a, logp, v = resp_q.get()
            responses[req_id] = (a, logp, v)
            got += 1
        TIMERS['queue_wait'] += time.perf_counter() - t0
        # 3) 应用并 step
        nxt = []
        for g, acts in zip(games, local_actions):
            for req_id, (name, seat, o, vec) in g.pending.items():
                a, logp, v = responses[req_id]
                acts[name] = a
                mid, argmax, collect = g.assignment[seat]
                if collect and g.mode != 'eval':
                    g.traj[seat].append(
                        (o['observation'], vec, o['action_mask'], a, logp, v))
            g.pending.clear()
            if g.logger:
                g.logger.pre_step(acts)
            t0 = time.perf_counter()
            obs, rew, done = g.env.step(acts)
            TIMERS['env_step'] += time.perf_counter() - t0
            if g.logger:
                g.logger.post_step([g.env.reward[i] if g.env.reward else 0
                                    for i in range(4)], done)
            if done:
                finish_game(g)
            else:
                g.cur_obs = obs
                nxt.append(g)
        games = nxt

        if time.time() - t_report > 30:
            stats_q.put(('timers', {'worker': worker_id, **TIMERS}))
            t_report = time.time()
        if log_games and len(logged_games) >= log_games and not eval_mode:
            import pickle
            with open(os.path.join(cfg.run_dir, 'logged_games_w%d.pkl' % worker_id), 'wb') as f:
                pickle.dump(logged_games, f)
            log_games = 0
