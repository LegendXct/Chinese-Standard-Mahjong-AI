"""M3 PPO 自对弈主程序(learner + 编排)。

模式:
  --mode gate-smoke   门禁1/2/4:N 局冒烟(纯 SL 权重,不更新),输出零和/胡牌率/吞吐/计时拆分,
                      worker0 额外记录 5 局牌谱+轨迹供一致性抽查
  --mode gate-zero    门禁3:收满 1 个 rollout,lr=0 更新验证权重不变 + KL(π‖π_SL) +
                      真实 lr 首次更新的 ratio/clip 统计
  --mode pilot        试点(默认 max_hours=4)
  --mode main         主跑(--hours 覆盖,评测节奏放宽到 120min/500 组)

用法:
  python src/rl/train_rl.py --mode pilot --set max_hours=3
"""
import argparse
import csv
import json
import math
import os
import pickle
import queue
import random
import shutil
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

_SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _SRC)
from rl.config import RLConfig, parse_overrides  # noqa: E402
from rl.common import (dist_stats, kl_masked, sd_to_cpu_bf16,  # noqa: E402
                       TRAJ_SLOTS, TRAJ_SLOT_BYTES, read_traj)
from rl.infer_server import run_infer_server  # noqa: E402
from rl.env_worker import run_env_worker  # noqa: E402


class TrajCollector:
    """从共享内存 arena 取回轨迹并释放槽位。"""

    def __init__(self, shm, slot_sems):
        self.buf = shm.buf
        self.sems = slot_sems

    def collect(self, traj_q, grace=0.002, max_items=4000, copy=True):
        out = []
        n = 0
        while n < max_items:
            try:
                msg = traj_q.get(timeout=grace)
            except queue.Empty:
                break
            n += 1
            if msg[0] == 'traj':
                _, wid, slot, T = msg
                base = (wid * TRAJ_SLOTS + slot) * TRAJ_SLOT_BYTES
                if copy:
                    out.append(read_traj(self.buf[base:base + TRAJ_SLOT_BYTES], T))
                self.sems[wid].release()
            else:   # inline 退路
                out.append(msg[2])
        return out


def build_eval_tasks(cfg, groups, seed_tag, opponent='sl', kind='quick'):
    from arena.duplicate import make_wall
    tasks = []
    for g in range(groups):
        wall = make_wall((cfg.seed, 'eval', kind, seed_tag, g))
        wind = random.Random(str((cfg.seed, 'evalw', kind, seed_tag, g))).randrange(4)
        for s in range(4):
            asg = {i: (('eval_cand', True, False) if i == s else (opponent, True, False))
                   for i in range(4)}
            tasks.append((asg, wall, wind, {'group': g, 'cand_seat': s, 'kind': kind}))
    return tasks


def chunk_tasks(tasks, per_chunk=100):
    return [tasks[i:i + per_chunk] for i in range(0, len(tasks), per_chunk)]


def eval_summary(results):
    by_group = {}
    scores = []
    n_hu = n_dp = 0
    win_fans = []          # 候选和牌时的番(arena 胜局番)
    dealin_fans = []       # 候选点炮时(放给对方)的番
    for r in results:
        s = r['scores']
        cs = r['meta']['cand_seat']
        cand = s[cs]
        scores.append(cand)
        d = cand - (sum(s) - cand) / 3.0
        by_group.setdefault(r['meta']['group'], []).append(d)
        if max(s) > 0:
            win = int(np.argmax(s))
            if win == cs:
                n_hu += 1
            elif s[cs] == min(s) and sorted(x for i, x in enumerate(s) if i != win)[0] != \
                    sorted(x for i, x in enumerate(s) if i != win)[-1]:
                n_dp += 1
        w = r.get('win', -1)            # finish_game eval 分支带回(旧结果缺则跳过)
        if w >= 0 and w == cs:
            win_fans.append(r.get('fan', 0))
        if r.get('dealer_in', -1) == cs:
            dealin_fans.append(r.get('fan', 0))
    dg = [np.mean(v) for v in by_group.values()]
    n = len(results)
    return {'n': n, 'score_mean': float(np.mean(scores)),
            'pair_diff': float(np.mean(dg)),
            'pair_se': float(np.std(dg, ddof=1) / math.sqrt(len(dg))) if len(dg) > 1 else 0.0,
            'hu_rate': n_hu / n, 'dianpao_rate': n_dp / n,
            'win_fan_mean': float(np.mean(win_fans)) if win_fans else 0.0,
            'n_win': len(win_fans),
            'dealin_fan_mean': float(np.mean(dealin_fans)) if dealin_fans else 0.0,
            'n_dealin': len(dealin_fans),
            'dealin_big': int(sum(1 for f in dealin_fans if f >= 16)),
            'dealin_big_rate': float(sum(1 for f in dealin_fans if f >= 16)) / n}


def run_eval_once(cfg, infer_ctrl_q, eval_cmd_q, eval_res_q, learner, eval_opps):
    """--mode eval:对 cfg.init_model 跑一次完整评测(arena 胜局番 + 点炮分番),打印基线后退出。"""
    learner.push_weights(infer_ctrl_q, 'eval_cand')
    time.sleep(2)
    tasks = []
    for opp, kind in eval_opps:
        tasks += build_eval_tasks(cfg, cfg.eval_full_groups, 'measure', opp, kind)
    for ch in chunk_tasks(tasks):
        eval_cmd_q.put(('eval', ch))
    n_expect = len(tasks)
    results = []
    t0 = time.time()
    while len(results) < n_expect:
        got = drain(eval_res_q)
        if got:
            results.extend(got)
        else:
            time.sleep(0.1)
        if time.time() - t0 > 5400:
            print('[measure] TIMEOUT %d/%d' % (len(results), n_expect), flush=True)
            break
    print('[measure] 候选=%s 每对手%d组 收齐%d/%d' % (
        cfg.init_model, cfg.eval_full_groups, len(results), n_expect), flush=True)
    COMP = ('full_sl', 'full_bighand', 'full_def')
    comp_win_fans, comp_dealin_fans, n_comp = [], [], 0
    for opp, kind in eval_opps:
        sub = [r for r in results if r['meta']['kind'] == kind]
        if not sub:
            continue
        es = eval_summary(sub)
        print('[measure] %-11s pair %+.3f±%.3f hu %.3f dp %.3f | 胜局番 %.2f(n%d) '
              '点炮番 %.2f(n%d) ≥16点炮 %d=%.4f/局' % (
                  kind, es['pair_diff'], es['pair_se'], es['hu_rate'], es['dianpao_rate'],
                  es['win_fan_mean'], es['n_win'], es['dealin_fan_mean'], es['n_dealin'],
                  es['dealin_big'], es['dealin_big_rate']), flush=True)
        if kind in COMP:
            n_comp += es['n']
            for r in sub:
                cs = r['meta']['cand_seat']
                if r.get('win', -1) == cs and r.get('win', -1) >= 0:
                    comp_win_fans.append(r.get('fan', 0))
                if r.get('dealer_in', -1) == cs:
                    comp_dealin_fans.append(r.get('fan', 0))
    big = sum(1 for f in comp_dealin_fans if f >= 16)
    print('[measure] === 竞争子集 {M5,bighand,def} 基线(batch-2 对照) ===', flush=True)
    print('[measure]   胜局平均番 %.3f (n=%d 胜局)' % (
        (float(np.mean(comp_win_fans)) if comp_win_fans else 0.0), len(comp_win_fans)), flush=True)
    print('[measure]   点炮平均番 %.3f (n=%d 点炮) | ≥16番点炮 %d 次 = %.4f/局 (共%d局)' % (
        (float(np.mean(comp_dealin_fans)) if comp_dealin_fans else 0.0), len(comp_dealin_fans),
        big, (big / n_comp if n_comp else 0.0), n_comp), flush=True)


class Learner:

    def __init__(self, cfg, mode):
        if cfg.arch == 'v2':
            from model_v2 import MahjongNetV2 as NetClass
        else:
            from model_v1 import MahjongNetV1 as NetClass
        self.cfg = cfg
        self.mode = mode
        self.device = torch.device('cuda')
        self.model, _ = NetClass.load(cfg.init_model or cfg.sl_champion)
        self.model = self.model.to(self.device)
        self.sl_model, _ = NetClass.load(cfg.sl_champion)
        self.sl_model = self.sl_model.to(self.device).eval()
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
        self.opt = torch.optim.AdamW(
            [{'params': decay, 'weight_decay': cfg.wd},
             {'params': no_decay, 'weight_decay': 0.0}], lr=cfg.lr)
        self.n_updates = self.n_games = self.n_samples = 0
        self.pool_ids = []
        self.best_diff = -1e9
        self.entropy0 = None
        self.hu0 = None
        self.weight_version = 0
        # KL 再锚定协议状态:sl_model 为锚点(进损失);sl0_model 为原始 M5(纯监控)
        self.sl0_model, _ = NetClass.load(cfg.sl_champion)
        self.sl0_model = self.sl0_model.to(self.device).eval()
        self.reanchors = 0
        self.reanchor_events = []
        self.lr_escalations = 0
        self.saved_lr = None

    # ---------------- PPO ----------------
    def ppo_update(self, batch, lr_override=None):
        cfg = self.cfg
        dev = self.device
        obs = torch.from_numpy(batch['obs'])
        vec = torch.from_numpy(batch['vec'])
        mask = torch.from_numpy(batch['mask'])
        act = torch.from_numpy(batch['act'])
        logp_old = torch.from_numpy(batch['logp'])
        adv_all = torch.from_numpy(batch['adv'])
        ret = torch.from_numpy(batch['ret'])
        N = len(act)
        lr = lr_override if lr_override is not None else cfg.lr
        for g in self.opt.param_groups:
            g['lr'] = lr

        if cfg.arch == 'v2':
            from dataset_v2 import normalize_obs
        else:
            from dataset_v1 import normalize_obs
        stats = {k: 0.0 for k in ('policy', 'value', 'entropy', 'kl_sl', 'kl_sl0',
                                  'kl_old', 'clip_frac', 'ratio_mean', 'adv_std')}
        nmb = 0
        self.model.train()
        vcoef = cfg.value_coef
        if cfg.value_warmup_updates > 0 and self.n_updates < cfg.value_warmup_updates:
            vcoef = cfg.value_coef * cfg.value_warmup_mult   # fan backward 期 value head 多练
        for ep in range(cfg.ppo_epochs):
            perm = torch.randperm(N)
            for i0 in range(0, N - cfg.minibatch + 1, cfg.minibatch):
                idx = perm[i0:i0 + cfg.minibatch]
                o = normalize_obs(obs[idx].to(dev, non_blocking=True))
                vc = vec[idx].to(dev, non_blocking=True)
                m = mask[idx].to(dev, non_blocking=True).float()
                a = act[idx].to(dev, non_blocking=True)
                lp_old = logp_old[idx].to(dev, non_blocking=True)
                A = adv_all[idx].to(dev, non_blocking=True)
                R = ret[idx].to(dev, non_blocking=True)
                A = (A - A.mean()) / (A.std() + 1e-8)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, value = self.model(o, vc)
                    with torch.no_grad():
                        sl_logits, _ = self.sl_model(o, vc)
                        sl0_logits, _ = self.sl0_model(o, vc)
                logits = logits.float()
                value = value.float()
                lp_new, ent, logp_all = dist_stats(logits, m, a)
                _, _, logp_sl = dist_stats(sl_logits.float(), m)
                ratio = (lp_new - lp_old).exp()
                pl = -torch.min(ratio * A,
                                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
                vl = F.smooth_l1_loss(value, R)
                kl_sl = kl_masked(logp_all, logp_sl, m).mean()
                loss = pl + vcoef * vl + cfg.kl_sl_coef * kl_sl \
                    - cfg.entropy_coef * ent.mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()
                with torch.no_grad():
                    _, _, logp_sl0 = dist_stats(sl0_logits.float(), m)
                    stats['kl_sl0'] += kl_masked(logp_all, logp_sl0, m).mean().item()
                    stats['policy'] += pl.item()
                    stats['value'] += vl.item()
                    stats['entropy'] += ent.mean().item()
                    stats['kl_sl'] += kl_sl.item()
                    stats['kl_old'] += (lp_old - lp_new).mean().item()
                    stats['clip_frac'] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                    stats['ratio_mean'] += ratio.mean().item()
                    stats['adv_std'] += adv_all[idx].std().item()
                nmb += 1
        self.n_updates += 1
        return {k: v / max(nmb, 1) for k, v in stats.items()}

    def push_weights(self, ctrl_q, model_id='cur'):
        self.weight_version += 1
        ctrl_q.put(('set', model_id, sd_to_cpu_bf16(self.model.state_dict()),
                    self.weight_version))

    def checkpoint(self, path, extra=None):
        payload = {
            'config': self.model.config, 'state_dict': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'counters': {'updates': self.n_updates, 'games': self.n_games,
                         'samples': self.n_samples, 'weight_version': self.weight_version,
                         'best_diff': self.best_diff, 'entropy0': self.entropy0,
                         'hu0': self.hu0},
            'pool_ids': self.pool_ids,
            'anchor_state_dict': self.sl_model.state_dict() if self.reanchors else None,
            'reanchors': self.reanchors,
            'reanchor_events': self.reanchor_events,
            'lr_now': self.cfg.lr,
            'lr_escalations': getattr(self, 'lr_escalations', 0),
        }
        if extra:
            payload.update(extra)
        tmp = path + '.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def restore(self, path):
        p = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(p['state_dict'])
        self.model = self.model.to(self.device)
        self.opt.load_state_dict(p['optimizer'])
        c = p['counters']
        self.n_updates, self.n_games, self.n_samples = c['updates'], c['games'], c['samples']
        self.weight_version = c['weight_version']
        self.best_diff = c['best_diff']
        self.entropy0 = c.get('entropy0')
        self.hu0 = c.get('hu0')
        self.pool_ids = p['pool_ids']
        self.reanchors = p.get('reanchors', 0)
        self.reanchor_events = p.get('reanchor_events', [])
        self.lr_escalations = p.get('lr_escalations', 0)
        self.saved_lr = p.get('lr_now')
        if p.get('anchor_state_dict'):
            self.sl_model.load_state_dict(p['anchor_state_dict'])
            self.sl_model = self.sl_model.to(self.device).eval()
        return p

    def reanchor(self, ckpt_path, eval_info):
        """KL 再锚定:锚点替换为 eval-best 权重,事件入账。"""
        p = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.sl_model.load_state_dict({k: v.float() for k, v in p['state_dict'].items()})
        self.sl_model = self.sl_model.to(self.device).eval()
        self.reanchors += 1
        ev = {'time': time.strftime('%F %T'), 'update': self.n_updates,
              'n_reanchor': self.reanchors, 'eval': eval_info}
        self.reanchor_events.append(ev)
        print('[REANCHOR #%d] anchor <- eval_best (upd %d, eval %s)' % (
            self.reanchors, self.n_updates, eval_info), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['gate-smoke', 'gate-zero', 'pilot', 'main', 'eval'],
                    required=True)
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke-games', type=int, default=200)
    args = ap.parse_args()

    cfg = parse_overrides(RLConfig(), args.set)
    if args.mode == 'main':
        cfg.first_hour_gate = True       # 快评 30min/200 组 + 完整评测 120min/500 组
    else:
        cfg.eval_full_min = 1e9          # 试点/门禁不跑完整评测
    os.makedirs(cfg.run_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.run_dir, 'pool'), exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    import multiprocessing as mp
    from multiprocessing import shared_memory
    ctx = mp.get_context('spawn')
    req_q = ctx.Queue(maxsize=8192)
    n_w = cfg.n_env_workers
    n_e = cfg.n_eval_workers
    resp_qs = {i: ctx.Queue() for i in range(n_w + n_e)}
    traj_q = ctx.Queue()                                 # 只走小描述符,无须限长
    stats_q = ctx.Queue(maxsize=8192)
    infer_ctrl_q = ctx.Queue()
    eval_cmd_q = ctx.Queue()
    eval_res_q = ctx.Queue()
    mgr = ctx.Manager()
    # 固定陪练:bighand/def 走 v2 infer_server;rl_v1b 是 v1 模型,worker 本地推理
    init_models = {'cur': cfg.sl_champion, 'sl': cfg.sl_champion,
                   'eval_cand': cfg.sl_champion}
    fixed_ids = ['sl']
    eval_opps = [('sl', 'full_sl')]
    for sid, spath, sw in (
            ('bighand', os.path.join(cfg.spar_dir, 'spar_bighand.pt'), cfg.w_bighand),
            ('def', os.path.join(cfg.spar_dir, 'spar_def.pt'), cfg.w_def)):
        if sw > 0 and os.path.exists(spath):
            init_models[sid] = spath
            fixed_ids.append(sid)
            eval_opps.append((sid, 'full_' + sid))
        elif sw > 0:
            print('[pool] WARN sparring %s w=%.0f 但缺 ckpt: %s' % (sid, sw, spath), flush=True)
    if cfg.w_rl_v1b > 0 and os.path.exists(cfg.rl_v1b_ckpt):
        fixed_ids.append('rl_v1b')                    # worker 本地 v1 推理
        eval_opps.append(('rl_v1b', 'full_rlv1b'))
    eval_opps.append(('rulebot', 'full_rule'))
    print('[pool] fixed=%s eval_opps=%s' % (fixed_ids, [k for _, k in eval_opps]), flush=True)
    ctrl = mgr.dict({'snapshots': [], 'fixed': fixed_ids, 'stop': False,
                     'reward_alpha': cfg.reward_alpha, 'reward_beta': cfg.reward_beta})
    shm = shared_memory.SharedMemory(
        create=True, size=(n_w + n_e) * TRAJ_SLOTS * TRAJ_SLOT_BYTES)
    slot_sems = [ctx.BoundedSemaphore(TRAJ_SLOTS) for _ in range(n_w + n_e)]
    collector = TrajCollector(shm, slot_sems)

    learner = Learner(cfg, args.mode)
    if args.resume and os.path.exists(os.path.join(cfg.run_dir, 'last.pt')):
        cli_lr = any(s.startswith('lr=') for s in (args.set or []))
        p = learner.restore(os.path.join(cfg.run_dir, 'last.pt'))
        ctrl['snapshots'] = list(learner.pool_ids)
        if cli_lr and learner.saved_lr and abs(learner.saved_lr - cfg.lr) > 1e-12:
            print('[LR_RESTORE] %s: checkpoint lr %.2e -> cli lr %.2e' % (
                time.strftime('%F %T'), learner.saved_lr, cfg.lr), flush=True)
        elif not cli_lr and learner.saved_lr:
            cfg.lr = learner.saved_lr
        print('[resume] updates=%d games=%d samples=%d pool=%d lr=%.2e' % (
            learner.n_updates, learner.n_games, learner.n_samples,
            len(learner.pool_ids), cfg.lr), flush=True)

    procs = []
    pi = ctx.Process(target=run_infer_server, args=(cfg, init_models, req_q, resp_qs,
                                                    infer_ctrl_q, stats_q), daemon=True)
    pi.start()
    procs.append(pi)
    log_games = 5 if args.mode == 'gate-smoke' else 0
    for i in range(n_w):
        pw = ctx.Process(target=run_env_worker,
                         args=(i, cfg, req_q, resp_qs[i], traj_q, ctrl, stats_q,
                               shm.name, slot_sems[i],
                               None, None, log_games if i == 0 else 0), daemon=True)
        pw.start()
        procs.append(pw)
    for j in range(n_e):
        pe = ctx.Process(target=run_env_worker,
                         args=(n_w + j, cfg, req_q, resp_qs[n_w + j], traj_q, ctrl, stats_q,
                               shm.name, slot_sems[n_w + j],
                               eval_cmd_q, eval_res_q, 0), daemon=True)
        pe.start()
        procs.append(pe)
    if args.resume:
        # 把池快照与当前权重重新注册进推理服务器
        for mid in learner.pool_ids:
            sd, _ = _load_pool_sd(cfg, mid)
            infer_ctrl_q.put(('set', mid, sd, 0))
        learner.push_weights(infer_ctrl_q, 'cur')

    try:
        if args.mode == 'gate-smoke':
            run_gate_smoke(cfg, args, ctrl, traj_q, stats_q, learner, collector)
        elif args.mode == 'gate-zero':
            run_gate_zero(cfg, ctrl, traj_q, stats_q, learner, infer_ctrl_q, collector)
        elif args.mode == 'eval':
            run_eval_once(cfg, infer_ctrl_q, eval_cmd_q, eval_res_q, learner, eval_opps)
        else:
            run_training(cfg, args, ctrl, traj_q, stats_q, learner, infer_ctrl_q,
                         eval_cmd_q, eval_res_q, collector, eval_opps)
    except KeyboardInterrupt:
        print('[SIGINT] graceful stop: saving checkpoint...', flush=True)
        learner.checkpoint(os.path.join(cfg.run_dir, 'last.pt'))
        print('[SIGINT] checkpoint saved (upd %d)' % learner.n_updates, flush=True)
    finally:
        ctrl['stop'] = True
        time.sleep(2)
        for p in procs:
            p.terminate()
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def _load_pool_sd(cfg, mid):
    path = os.path.join(cfg.run_dir, 'pool', mid + '.pt')
    p = torch.load(path, map_location='cpu', weights_only=True)
    return p['state_dict'], p


def drain(q, grace=0.002, max_items=4000):
    """排空队列。大对象(轨迹 ~210KB)跨多个管道块传输,get_nowait 会在
    feeder 换块间隙误报 Empty,导致读端远慢于写端、队列长期满载、
    worker 全体阻塞在 put(实测 57s 后吞吐崖跌 5 倍)。
    用小宽限的阻塞 get 桥接间隙;max_items 防止持续高产时读不返回。"""
    out = []
    while len(out) < max_items:
        try:
            out.append(q.get(timeout=grace))
        except queue.Empty:
            return out
    return out


class StatsAgg:
    def __init__(self):
        self.games = deque(maxlen=2000)
        self.timers = {}
        self.infer = {}

    def feed(self, items):
        for kind, d in items:
            if kind == 'game':
                self.games.append(d)
            elif kind == 'timers':
                self.timers[d['worker']] = d
            elif kind == 'infer':
                self.infer = d

    def game_stats(self):
        g = list(self.games)
        if not g:
            return {}
        hu = np.mean([x['hu'] for x in g])
        fans = [x['fan'] for x in g if x['hu']]
        return {'hu_rate': float(hu), 'huang_rate': float(1 - hu),
                'avg_fan': float(np.mean(fans)) if fans else 0.0,
                'zero_sum_violation': sum(1 for x in g if sum(x['scores']) != 0),
                'invalid': sum(1 for x in g if max(x['scores']) == 10)}


def run_gate_smoke(cfg, args, ctrl, traj_q, stats_q, learner, collector):
    agg = StatsAgg()
    n_games = 0
    n_samples = 0
    t0 = time.time()
    target = args.smoke_games
    warm_mark = None   # (t, games, samples) at warmup end,稳态斜率 = 之后的增量
    n_drained = 0
    while n_games < target:
        n_drained += len(collector.collect(traj_q, copy=False))  # 样本数从 game 统计取
        items = drain(stats_q)
        agg.feed(items)
        for k, d in items:
            if k == 'game':
                n_games += 1
                n_samples += d['n_steps']
        if warm_mark is None and time.time() - t0 > 45:
            warm_mark = (time.time(), n_games, n_samples)
        if not hasattr(run_gate_smoke, '_win'):
            run_gate_smoke._win = (time.time(), 0, dict())
        tw, gw, timw = run_gate_smoke._win
        if time.time() - tw > 30:
            cur_tim = {k: sum(t.get(k, 0) for t in agg.timers.values())
                       for k in ('env_step', 'render', 'shanten', 'vec', 'queue_wait',
                                 'submit_put', 'reset', 'ctrl_ipc', 'finish_put')}
            d_tim = {k: round(cur_tim[k] - timw.get(k, 0), 1) for k in cur_tim}
            try:
                qsz = traj_q.qsize()
            except NotImplementedError:
                qsz = -1
            print('[win] %.0fs: %.2f games/s | drained=%d qsize=%d | timer-delta %s | infer %s' % (
                time.time() - t0, (n_games - gw) / (time.time() - tw),
                n_drained, qsz, json.dumps(d_tim), json.dumps(agg.infer)), flush=True)
            n_drained = 0
            run_gate_smoke._win = (time.time(), n_games, cur_tim)
        time.sleep(0.2)
    dt = time.time() - t0
    if warm_mark and n_games > warm_mark[1]:
        sdt = time.time() - warm_mark[0]
        print('steady-state (after 45s warmup): %.2f games/s, %.0f samples/s over %.0fs'
              % ((n_games - warm_mark[1]) / sdt, (n_samples - warm_mark[2]) / sdt, sdt),
              flush=True)
    gs = agg.game_stats()
    total_t = {k: sum(t.get(k, 0) for t in agg.timers.values())
               for k in ('env_step', 'render', 'shanten', 'vec', 'queue_wait')}
    # 注:env_step⊃render⊃shanten、vec 亦含 shanten,桶间重叠;vec 绝对秒数 / (worker×elapsed)
    # 才是 vec 占 worker 墙钟的真实比例(pct 仅作粗参考)
    cpu_total = sum(total_t.values()) or 1
    print(json.dumps({
        'games': n_games, 'samples': n_samples, 'elapsed_s': round(dt, 1),
        'games_per_s': round(n_games / dt, 2), 'samples_per_s': round(n_samples / dt, 1),
        'game_stats': gs,
        'actor_time_split': {k: round(v, 1) for k, v in total_t.items()},
        'actor_time_pct': {k: round(100 * v / cpu_total, 1) for k, v in total_t.items()},
        'note': 'render 含 shanten;env_step 含 render',
        'infer': agg.infer,
    }, ensure_ascii=False, indent=1), flush=True)
    print('GATE_SMOKE_DONE')


def run_gate_zero(cfg, ctrl, traj_q, stats_q, learner, infer_ctrl_q, collector):
    buf = []
    n = 0
    while n < cfg.rollout_size:
        for it in collector.collect(traj_q):
            buf.append(it)
            n += len(it['act'])
        drain(stats_q)
        time.sleep(0.1)
    batch = {k: np.concatenate([b[k] for b in buf]) for k in buf[0]}
    print('rollout collected: %d samples' % len(batch['act']), flush=True)

    before = {k: v.detach().clone() for k, v in learner.model.state_dict().items()}
    st0 = learner.ppo_update(batch, lr_override=0.0)
    changed = [k for k, v in learner.model.state_dict().items()
               if not torch.equal(v, before[k])]
    print('zero-lr update: changed tensors = %d (expect 0)' % len(changed), flush=True)
    print('initial KL(pi||pi_SL) = %.3e (expect <1e-6)' % st0['kl_sl'], flush=True)
    print('zero-lr ratio_mean = %.6f clip_frac = %.4f' % (
        st0['ratio_mean'], st0['clip_frac']), flush=True)
    st1 = learner.ppo_update(batch)
    print('real-lr first update: ratio_mean = %.4f (expect~1) clip_frac = %.4f (expect<0.02) '
          'kl_sl = %.3e' % (st1['ratio_mean'], st1['clip_frac'], st1['kl_sl']), flush=True)
    ok = (len(changed) == 0 and st0['kl_sl'] < 1e-6
          and abs(st1['ratio_mean'] - 1) < 0.05 and st1['clip_frac'] < 0.02)
    print('GATE_ZERO_%s' % ('PASS' if ok else 'FAIL'))


def run_training(cfg, args, ctrl, traj_q, stats_q, learner, infer_ctrl_q,
                 eval_cmd_q, eval_res_q, collector, eval_opps):
    agg = StatsAgg()
    run_t0 = time.time()
    buf, buf_n = [], 0
    last_log = last_eval = last_full = last_summary = last_snap = time.time()
    eval_pending = None      # {'kind','n_expect','results',...};quick 与 full 互斥排队
    quick_history = []
    full_sl_history = []     # 完整评测 vs SL,eval-best 与再锚定的依据
    thru_window = deque(maxlen=8)
    alarm = None
    gate1h = {'done': not cfg.first_hour_gate, 'kl_start': None, 'ent_start': None}
    kl_now = 0.0
    last_st = {}
    best_time = time.time()   # 完整评测连续 12h 无改善 -> 早停
    clip_hist = deque()       # (t, clip_frac) 供阶梯控制器取 6h 窗口中位数
    vloss_hist = deque()
    lr_hold_until = 0.0       # 降档后至少保持 6h

    mcsv = open(os.path.join(cfg.run_dir, 'train_metrics.csv'), 'a', newline='')
    mw = csv.writer(mcsv)
    if mcsv.tell() == 0:
        mw.writerow(['time_min', 'update', 'games', 'samples', 'policy', 'value', 'entropy',
                     'kl_sl', 'kl_old', 'clip_frac', 'ratio_mean', 'adv_std',
                     'games_per_s', 'samples_per_s', 'hu_rate', 'huang_rate', 'avg_fan',
                     'infer_batch', 'pool_size'])
    ecsv = open(os.path.join(cfg.run_dir, 'eval_log.csv'), 'a', newline='')
    ew = csv.writer(ecsv)
    if ecsv.tell() == 0:
        ew.writerow(['time_min', 'update', 'games', 'pair_diff', 'pair_se',
                     'score_mean', 'hu_rate', 'n'])
    effc = os.path.join(cfg.run_dir, 'eval_full.csv')
    if not os.path.exists(effc) or os.path.getsize(effc) == 0:
        with open(effc, 'a') as ff:
            ff.write('time_min,update,games,comp_mean,comp_min,gen_mean,dianpao,'
                     'win_fan,big_dp16,'
                     + ','.join(k for _, k in eval_opps) + '\n')

    t_prev = time.time()
    s_prev = learner.n_samples
    g_prev = learner.n_games
    while (time.time() - run_t0 < cfg.max_hours * 3600
           and (cfg.max_updates <= 0 or learner.n_updates < cfg.max_updates)
           and not alarm):
        for it in collector.collect(traj_q):
            buf.append(it)
            buf_n += len(it['act'])
            learner.n_samples += len(it['act'])
        items = drain(stats_q)
        agg.feed(items)
        learner.n_games += sum(1 for k, _ in items if k == 'game')

        if buf_n >= cfg.rollout_size:
            batch = {k: np.concatenate([b[k] for b in buf]) for k in buf[0]}
            buf, buf_n = [], 0
            st = learner.ppo_update(batch)
            learner.push_weights(infer_ctrl_q, 'cur')
            gs = agg.game_stats()
            if learner.entropy0 is None:
                learner.entropy0 = st['entropy']
                learner.hu0 = gs.get('hu_rate', 0.5)
            dt = time.time() - t_prev
            gps = (learner.n_games - g_prev) / dt
            sps = (learner.n_samples - s_prev) / dt
            t_prev, g_prev, s_prev = time.time(), learner.n_games, learner.n_samples
            thru_window.append(gps)
            mw.writerow(['%.1f' % ((time.time() - run_t0) / 60), learner.n_updates,
                         learner.n_games, learner.n_samples,
                         '%.4f' % st['policy'], '%.4f' % st['value'],
                         '%.4f' % st['entropy'], '%.4f' % st['kl_sl'],
                         '%.5f' % st['kl_old'], '%.4f' % st['clip_frac'],
                         '%.4f' % st['ratio_mean'], '%.3f' % st['adv_std'],
                         '%.2f' % gps, '%.0f' % sps,
                         '%.4f' % gs.get('hu_rate', 0), '%.4f' % gs.get('huang_rate', 0),
                         '%.2f' % gs.get('avg_fan', 0),
                         '%.1f' % agg.infer.get('avg_batch', 0), len(learner.pool_ids) + 1])
            mcsv.flush()
            kl_now = st['kl_sl']
            last_st = st
            clip_hist.append((time.time(), st['clip_frac']))
            vloss_hist.append((time.time(), st['value']))
            while clip_hist and time.time() - clip_hist[0][0] > 13 * 3600:
                clip_hist.popleft()
                vloss_hist.popleft()
            with open(os.path.join(cfg.run_dir, 'monitor.csv'), 'a') as mf:
                mf.write('%.1f,%d,%.4f,%.4f,%d\n' % (
                    (time.time() - run_t0) / 60, learner.n_updates,
                    st['kl_sl'], st['kl_sl0'], learner.reanchors))
            if gate1h['kl_start'] is None:
                gate1h['kl_start'] = st['kl_sl']
                gate1h['ent_start'] = st['entropy']

            # ---- KL 再锚定协议(替代原 KL 硬告警)----
            if st['kl_sl'] >= cfg.alarm_kl_sl:
                ok_eval = bool(full_sl_history and full_sl_history[-1]['comp_mean'] > 0)
                if ok_eval and learner.reanchors < cfg.kl_reanchor_max:
                    learner.reanchor(os.path.join(cfg.run_dir, 'eval_best.pt'),
                                     {'kl_at_trigger': round(st['kl_sl'], 4),
                                      'comp_mean': full_sl_history[-1]['comp_mean']})
                else:
                    alarm = ('KL %.3f >= %.2f, reanchors=%d, last_comp_mean=%s -> pause'
                             % (st['kl_sl'], cfg.alarm_kl_sl, learner.reanchors,
                                full_sl_history[-1]['comp_mean'] if full_sl_history else None))

            # 其余告警线
            if st['entropy'] < learner.entropy0 * cfg.alarm_entropy_frac:
                alarm = 'entropy %.3f < %.0f%% of initial %.3f' % (
                    st['entropy'], cfg.alarm_entropy_frac * 100, learner.entropy0)
            hu = gs.get('hu_rate', None)
            if hu is not None and learner.hu0 and hu < learner.hu0 * (1 - cfg.alarm_hu_drop):
                alarm = 'hu_rate %.3f dropped >%.0f%% from %.3f' % (
                    hu, cfg.alarm_hu_drop * 100, learner.hu0)
            if len(thru_window) == thru_window.maxlen and \
                    np.mean(list(thru_window)[-3:]) < np.mean(list(thru_window)[:3]) * cfg.alarm_thru_drop:
                alarm = 'throughput halved: %.2f -> %.2f games/s' % (
                    np.mean(list(thru_window)[:3]), np.mean(list(thru_window)[-3:]))

            if learner.n_updates % cfg.ckpt_every_updates == 0:
                learner.checkpoint(os.path.join(cfg.run_dir, 'last.pt'))
            if learner.n_updates % 200 == 0:
                learner.checkpoint(os.path.join(cfg.run_dir, 'upd_%06d.pt' % learner.n_updates))
                _prune_archives(cfg.run_dir, keep=20)

        now = time.time()
        if now - last_snap > cfg.pool_snapshot_min * 60:
            last_snap = now
            mid = 'pool_%05d' % learner.n_updates
            sd = sd_to_cpu_bf16(learner.model.state_dict())
            torch.save({'state_dict': sd}, os.path.join(cfg.run_dir, 'pool', mid + '.pt'))
            learner.pool_ids.append(mid)
            if len(learner.pool_ids) > cfg.pool_max:
                old = learner.pool_ids.pop(0)
                infer_ctrl_q.put(('drop', old))
                try:
                    os.remove(os.path.join(cfg.run_dir, 'pool', old + '.pt'))
                except OSError:
                    pass
            infer_ctrl_q.put(('set', mid, sd, learner.weight_version))
            ctrl['snapshots'] = list(learner.pool_ids)
            print('[pool] snapshot %s (snaps %d + fixed %d)' % (
                mid, len(learner.pool_ids), len(ctrl['fixed'])), flush=True)

        # ---- 评测调度:完整评测(2h)优先于快评(30min),同一时刻只跑一个 ----
        if eval_pending is None:
            if now - last_full > cfg.eval_full_min * 60:
                last_full = now
                last_eval = now   # 完整评测顺带覆盖一次快评窗口
                learner.push_weights(infer_ctrl_q, 'eval_cand')
                tasks = []
                for opp, kind in eval_opps:        # 泛化对手集 {M5,bighand,def,rl_v1b,RuleBot}
                    tasks += build_eval_tasks(cfg, cfg.eval_full_groups,
                                              learner.n_updates, opp, kind)
                for ch in chunk_tasks(tasks):
                    eval_cmd_q.put(('eval', ch))
                eval_pending = {'kind': 'full', 'n_expect': len(tasks), 'results': [],
                                'update': learner.n_updates}
            elif now - last_eval > cfg.eval_every_min * 60:
                last_eval = now
                learner.push_weights(infer_ctrl_q, 'eval_cand')
                tasks = build_eval_tasks(cfg, cfg.eval_groups, learner.n_updates,
                                         'sl', 'quick')
                for ch in chunk_tasks(tasks):
                    eval_cmd_q.put(('eval', ch))
                eval_pending = {'kind': 'quick', 'n_expect': len(tasks), 'results': [],
                                'update': learner.n_updates}
        if eval_pending is not None:
            for r in drain(eval_res_q):
                eval_pending['results'].append(r)
            if len(eval_pending['results']) >= eval_pending['n_expect']:
                tmin = (now - run_t0) / 60
                if eval_pending['kind'] == 'quick':
                    es = eval_summary(eval_pending['results'])
                    quick_history.append(es)
                    ew.writerow(['%.1f' % tmin, eval_pending['update'], learner.n_games,
                                 '%.3f' % es['pair_diff'], '%.3f' % es['pair_se'],
                                 '%.3f' % es['score_mean'], '%.4f' % es['hu_rate'], es['n']])
                    ecsv.flush()
                    print('[eval-quick] upd %d: pair_diff %.3f±%.3f hu %.3f' % (
                        eval_pending['update'], es['pair_diff'], es['pair_se'],
                        es['hu_rate']), flush=True)
                else:
                    rs = eval_pending['results']
                    per = {}
                    for opp, kind in eval_opps:
                        sub = [r for r in rs if r['meta']['kind'] == kind]
                        if sub:
                            per[kind] = eval_summary(sub)
                    diffs = [per[k]['pair_diff'] for k in per]
                    gen_mean = float(np.mean(diffs)) if diffs else 0.0
                    min_diff = float(np.min(diffs)) if diffs else 0.0
                    # 竞争子集 {M5,bighand,def}:eval-best 以它选(剔除 RuleBot/rl_v1b 虚高,
                    # 否则选择器奖励"打菜更狠"而非"赢强者")
                    COMP = ('full_sl', 'full_bighand', 'full_def')
                    cper = [(per[k]['pair_diff'], per[k]['pair_se']) for k in COMP if k in per]
                    comp_mean = float(np.mean([d for d, _ in cper])) if cper else 0.0
                    comp_se = float(np.sqrt(np.mean([s * s for _, s in cper]) / len(cper))) \
                        if cper else 0.0
                    comp_min, comp_min_se = min(cper) if cper else (0.0, 0.0)
                    # 点炮率:候选 vs 异质 NN 对手(排除 RuleBot,口径贴近天梯)
                    dps = [per[k]['dianpao_rate'] for k in per if k != 'full_rule']
                    cur_dp = float(np.mean(dps)) if dps else 0.0
                    hus = [per[k]['hu_rate'] for k in per if k != 'full_rule']
                    cur_hu = float(np.mean(hus)) if hus else 0.0
                    # batch-2:竞争子集 arena 胜局番 + ≥16番点炮率(对照基线 13.65 / 0.0443)
                    cw = [(per[k]['win_fan_mean'], per[k]['n_win']) for k in COMP if k in per]
                    nwin = sum(n for _, n in cw)
                    comp_win_fan = (sum(f * n for f, n in cw) / nwin) if nwin else 0.0
                    cbig = sum(per[k].get('dealin_big', 0) for k in COMP if k in per)
                    cn = sum(per[k]['n'] for k in COMP if k in per)
                    comp_big_dp = (cbig / cn) if cn else 0.0
                    entry = {'t_min': round(tmin, 1), 'update': eval_pending['update'],
                             'comp_mean': comp_mean, 'comp_se': comp_se, 'comp_min': comp_min,
                             'gen_mean': gen_mean, 'min_diff': min_diff,
                             'dianpao': cur_dp, 'hu_rate': cur_hu,
                             'win_fan': comp_win_fan, 'big_dp16': comp_big_dp, 'per': per}
                    full_sl_history.append(entry)
                    with open(os.path.join(cfg.run_dir, 'eval_full.csv'), 'a') as ff:
                        cols = ['%.1f' % tmin, str(eval_pending['update']),
                                str(learner.n_games), '%.3f' % comp_mean,
                                '%.3f' % comp_min, '%.3f' % gen_mean, '%.4f' % cur_dp,
                                '%.2f' % comp_win_fan, '%.4f' % comp_big_dp]
                        for _, kind in eval_opps:
                            cols.append('%.3f' % per[kind]['pair_diff']
                                        if kind in per else 'na')
                        ff.write(','.join(cols) + '\n')
                    print('[eval-FULL] upd %d: comp_mean %.3f (min %.3f) gen_mean %.3f '
                          'hu %.3f dp %.3f | win_fan %.2f big_dp16 %.4f | %s'
                          % (eval_pending['update'], comp_mean, comp_min, gen_mean,
                             cur_hu, cur_dp, comp_win_fan, comp_big_dp,
                             ' '.join('%s %.2f' % (k.replace('full_', ''),
                                                   per[k]['pair_diff']) for k in per)),
                          flush=True)
                    # eval-best 以竞争子集均值选 + 地板(最差竞争对手 >= -1×SE)
                    floor_ok = comp_min >= -cfg.eval_floor_se * comp_min_se
                    if comp_mean > learner.best_diff and floor_ok:
                        learner.best_diff = comp_mean
                        best_time = now
                        learner.checkpoint(os.path.join(cfg.run_dir, 'eval_best.pt'),
                                           extra={'eval': entry})
                        print('[eval-best] comp_mean updated: %.3f (min %.3f ok)'
                              % (comp_mean, comp_min), flush=True)
                    elif comp_mean > learner.best_diff:
                        print('[eval-best] comp_mean %.3f>best 但地板不过(min %.3f<-%.3f)跳过'
                              % (comp_mean, comp_min, cfg.eval_floor_se * comp_min_se), flush=True)
                    # 点炮红线:超 baseline+3pp 自动压 α 升 β(仅主跑,陪练短训不动)
                    if cfg.max_updates <= 0 and dps:
                        if not hasattr(run_training, '_dp_base'):
                            run_training._dp_base = cur_dp
                        elif cur_dp > run_training._dp_base + cfg.alarm_dianpao_pp:
                            na = max(cfg.reward_alpha * 0.8, 0.0)
                            nb = cfg.reward_beta * 1.2
                            cfg.reward_alpha, cfg.reward_beta = na, nb
                            ctrl['reward_alpha'], ctrl['reward_beta'] = na, nb
                            print('[REDLINE] dianpao %.3f > base %.3f +%.0fpp '
                                  '-> alpha=%.3f beta=%.3f' % (
                                      cur_dp, run_training._dp_base,
                                      cfg.alarm_dianpao_pp * 100, na, nb), flush=True)
                eval_pending = None
        if cfg.eval_full_min * 60 < 12 * 3600 and now - best_time > 12 * 3600:
            print('EARLY_STOP: no eval-best improvement for 12h', flush=True)
            break

        # ---- 启动后 1h 快照(只记录,不动作;"过热"才降 lr 的语义由阶梯控制器执行)----
        if not gate1h['done'] and now - run_t0 > 3600 and last_st:
            gate1h['done'] = True
            kl_slope = (kl_now - (gate1h['kl_start'] or 0)) / ((now - run_t0) / 3600)
            print('[snapshot1h] clip=%.4f kl_slope=%.4f/h entropy=%.4f (start %.4f) '
                  'last_quick=%s lr=%.2e' % (
                      last_st['clip_frac'], kl_slope, last_st['entropy'],
                      gate1h['ent_start'] or -1,
                      quick_history[-1] if quick_history else None, cfg.lr), flush=True)

        # ---- 6h 自包含摘要块 + 学习率阶梯控制器 ----
        if now - last_summary > 6 * 3600:
            last_summary = now
            gs6 = agg.game_stats()
            win = [c for t, c in clip_hist if now - t <= 6 * 3600]
            med_clip = float(np.median(win)) if win else 0.0
            ent_ok50 = last_st.get('entropy', 1) >= 0.5 * (learner.entropy0 or 1)
            lf = full_sl_history[-1] if full_sl_history else None
            sig_neg = bool(lf and lf['comp_mean'] < 0)   # 竞争子集均值转负
            v_now = [v for t, v in vloss_hist if now - t <= 6 * 3600]
            v_prev = [v for t, v in vloss_hist if 6 * 3600 < now - t <= 12 * 3600]
            v_diverge = bool(v_now and v_prev and
                             np.median(v_now) > 2 * np.median(v_prev)
                             and np.median(v_now) > 0.3)
            healthy = ent_ok50 and not sig_neg and not v_diverge
            two_no_sig_pos = (len(full_sl_history) >= 2 and
                              all(h['comp_mean'] < 2 * h['comp_se']
                                  for h in full_sl_history[-2:]))
            decision = 'HOLD'
            reason = 'med_clip=%.4f healthy=%s' % (med_clip, healthy)
            if (med_clip > 0.12 or not ent_ok50 or sig_neg or v_diverge) \
                    and now >= lr_hold_until:
                old = cfg.lr
                cfg.lr = max(cfg.lr * 0.5, 1e-5)
                lr_hold_until = now + 6 * 3600
                decision = 'LR_DEESCALATE %.2e->%.2e' % (old, cfg.lr)
                reason = 'med_clip=%.4f ent_ok=%s sig_neg=%s v_div=%s' % (
                    med_clip, ent_ok50, sig_neg, v_diverge)
            elif healthy and med_clip < 0.008 and two_no_sig_pos \
                    and learner.lr_escalations < 3:
                old = cfg.lr
                cfg.lr = min(cfg.lr * 1.5, 1e-4)
                if cfg.lr > old:
                    learner.lr_escalations += 1
                    decision = 'LR_ESCALATE #%d %.2e->%.2e' % (
                        learner.lr_escalations, old, cfg.lr)
                    reason = 'healthy, med_clip=%.4f<0.8%%, last2 full evals no sig-pos' \
                        % med_clip
            lf_s = ('comp=%.3f(min%.3f) gen=%.3f hu=%.3f dp=%.3f' % (
                lf['comp_mean'], lf['comp_min'], lf['gen_mean'], lf['hu_rate'],
                lf['dianpao'])) if lf else 'none'
            print('===== SUMMARY @%.1fh =====\n'
                  'updates=%d games=%d samples=%d reanchors=%d snaps=%d\n'
                  'last_full: %s | reward alpha=%.3f beta=%.3f\n'
                  'train: kl_anchor=%.4f kl_sl0=%.4f entropy=%.4f clip=%.4f hu=%.3f\n'
                  'resources: %.1f games/s | lr=%.2e win_med_clip=%.4f\n'
                  'lr_ladder: %s (%s)\n'
                  '==========================' % (
                      (now - run_t0) / 3600, learner.n_updates, learner.n_games,
                      learner.n_samples, learner.reanchors, len(learner.pool_ids),
                      lf_s, cfg.reward_alpha, cfg.reward_beta,
                      last_st.get('kl_sl', 0), last_st.get('kl_sl0', 0),
                      last_st.get('entropy', 0), last_st.get('clip_frac', 0),
                      gs6.get('hu_rate', 0),
                      np.mean(list(thru_window)) if thru_window else 0,
                      cfg.lr, med_clip, decision, reason), flush=True)

        if now - last_log > cfg.log_every_min * 60:
            last_log = now
            gs = agg.game_stats()
            print('[%.0fmin] upd %d games %d samples %d | hu %.3f | pool %d | %s' % (
                (now - run_t0) / 60, learner.n_updates, learner.n_games,
                learner.n_samples, gs.get('hu_rate', 0), len(learner.pool_ids) + 1,
                json.dumps(agg.infer)), flush=True)
        time.sleep(0.05)

    learner.checkpoint(os.path.join(cfg.run_dir, 'last.pt'))
    if alarm:
        print('!!!! ALARM: %s — training paused, checkpoint saved' % alarm, flush=True)
        print('TRAIN_ALARM')
        sys.exit(2)
    print('TRAIN_DONE (%.1f h, %d updates, %d games)' % (
        (time.time() - run_t0) / 3600, learner.n_updates, learner.n_games), flush=True)


def _prune_archives(run_dir, keep):
    arcs = sorted(f for f in os.listdir(run_dir) if f.startswith('upd_'))
    for f in arcs[:-keep]:
        os.remove(os.path.join(run_dir, f))


if __name__ == '__main__':
    main()
