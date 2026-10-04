"""单进程同步 PPO 的自对弈 worker(CPU,每进程 torch 单线程)。

每个 spawn 子进程常驻:一份冻结 M5(3 个对手座位用)+ 一份当前策略 π_θ(每轮由主进程
注入新权重)。每局自对弈轮换 1 个训练座位用 π_θ 采样并记录轨迹,另 3 座用冻结 M5(默认
argmax,与 arena 1v3 口径一致——目标就是"打赢 M5")。只收集训练座位样本。

轨迹单步记录 (obs_uint8, vec, mask_uint8, act, logp, value);终局零和得分/reward_scale 作
最终 reward(中间步 0),GAE 算 (adv, ret)。obs 与 vec 的口径严格对齐训练/arena/部署:
vec 用 feature_v2.compute_vec_from_obs 从原始 uint8 obs 现算,obs 喂模型前用
dataset_v2.normalize_obs 归一化。动作只在 action_mask 上采样,永不产生非法动作。
"""
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, '..', '..'))
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_W = {}   # 子进程常驻状态:torch / 冻结 M5 / 当前策略占位


def worker_init(m5_path):
    """spawn Pool initializer:常驻冻结 M5 与一份可覆盖权重的 π_θ 占位模型。"""
    import torch
    torch.set_num_threads(1)
    from model_v2 import MahjongNetV2
    frozen, _ = MahjongNetV2.load(m5_path, map_location='cpu')
    frozen.eval()
    cur, _ = MahjongNetV2.load(m5_path, map_location='cpu')   # 占位,每轮 load_state_dict 覆盖
    cur.eval()
    _W['torch'] = torch
    _W['frozen'] = frozen
    _W['cur'] = cur


def _act(net, obs_uint8, mask_uint8, sample):
    """单次决策。返回 (action:int, logp:float, value:float, vec:np.float32[107])。

    sample=True 多项式采样(训练座位),False argmax(对手)。obs 现算 vec、归一化喂模型。
    """
    torch = _W['torch']
    from dataset_v2 import normalize_obs
    from feature_v2 import compute_vec_from_obs
    from ppo import masked_log_softmax
    vec = compute_vec_from_obs(obs_uint8).astype(np.float32)
    with torch.no_grad():
        ot = normalize_obs(torch.from_numpy(np.ascontiguousarray(obs_uint8)).unsqueeze(0))
        vt = torch.from_numpy(vec).unsqueeze(0)
        mt = torch.from_numpy(np.ascontiguousarray(mask_uint8)).float().unsqueeze(0)
        logits, value = net(ot, vt)
        logp_all = masked_log_softmax(logits, mt)[0]
        if sample:
            a = int(torch.multinomial(logp_all.exp(), 1).item())
        else:
            a = int(logp_all.argmax().item())
        return a, float(logp_all[a].item()), float(value[0].item()), vec


def collect(cur_blob, n_games, run_seed, round_id, worker_id,
            gamma, lam, reward_scale, opp_sample,
            reward_mode='terminal', fanback_gamma=0.99, potential_coef=0.1):
    """跑 n_games 局自对弈,返回 (trajs, stats)。

    cur_blob: torch.save 出的 state_dict bytes(单对象传递,避免逐 tensor FD 化)。
    reward_mode:
      'terminal'  — Step 2:奖励仅终局 R=score/30(稀疏),其余步 0。
      'fanback'   — 把同一终局零和 R 沿轨迹从终局向前按 fanback_gamma 折扣、归一(Σw=1)再
                    分摊为逐步奖励(信用再分配,量级守恒,不改 R 的符号/总额);GAE 拿密集信号。
      'potential' — 终局 R 保留;另叠加势能塑形 c·F,F=gamma·Φ(s')−Φ(s),Φ=(6−shanten)/6∈[0,1]
                    (shanten 来自 vec[G_CUR_SHANTEN]/shanten_py),终局势能=0(Ng et al. 保策略不变)。
    价值头目标 ret 一律取与所用逐步奖励自洽的 GAE return-to-go(见 ppo.compute_gae_rewards)。
    trajs: list[dict]，每局训练座位一条,字段 obs/vec/mask/act/logp/adv/ret。
    """
    import io
    torch = _W['torch']
    from feature_v1 import FeatureAgentV1
    from env_fixed import MahjongGBEnvFixed
    from feature_v2 import G_CUR_SHANTEN
    from ppo import compute_gae, compute_gae_rewards

    cur = _W['cur']
    cur.load_state_dict(torch.load(io.BytesIO(cur_blob), map_location='cpu'))
    cur.eval()
    frozen = _W['frozen']

    seed_int = (int(run_seed) * 1000003 + int(round_id) * 1009 + int(worker_id) * 13) % (2 ** 31)
    random.seed(seed_int)        # env 牌墙 shuffle / 圈风 / 训练座位旋转 全部走全局 random
    np.random.seed(seed_int)

    stats = {'games': 0, 'samples': 0, 'wins': 0, 'huang': 0, 'illegal': 0,
             'final_scores': [], 'fans': [], 'steps': [], 'mask_assert_fail': 0,
             'shap_sum': 0.0, 'shap_absmean': 0.0, 'shap_n': 0}
    trajs = []
    env = MahjongGBEnvFixed(config={'agent_clz': FeatureAgentV1})

    for gi in range(n_games):
        train_seat = (worker_id + gi) % 4          # 逐局轮换训练座位,消座位偏置
        wind = random.randrange(4)
        obs = env.reset(prevalentWind=wind)        # 不传 tileWall -> 随机牌墙
        traj = []
        steps = 0
        done = False
        while not done:
            actions = {}
            for name, o in obs.items():
                seat = int(name.split('_')[1]) - 1
                mask = o['action_mask']
                legal = np.flatnonzero(mask)
                if len(legal) == 1:                # 唯一合法:直接执行,不入轨迹、不前向
                    actions[name] = int(legal[0])
                    continue
                if seat == train_seat:
                    a, logp, value, vec = _act(cur, o['observation'], mask, True)
                    if mask[a] != 1:               # 防御性断言:masked 采样应永远合法
                        stats['mask_assert_fail'] += 1
                    actions[name] = a
                    traj.append((o['observation'].copy(), vec, mask.copy(), a, logp, value))
                else:
                    a, _, _, _ = _act(frozen, o['observation'], mask, opp_sample)
                    actions[name] = a
            obs, _, done = env.step(actions)
            steps += 1
            if steps > 1000:                       # 安全阀(正常局远不及)
                break

        scores = [env.reward[i] if env.reward else 0 for i in range(4)]
        illegal = (max(scores) == 10 and min(scores) == -30)   # env 非法签名:违规 -30 其余 +10
        if illegal:
            stats['illegal'] += 1
        win = int(np.argmax(scores)) if (max(scores) > 0 and not illegal) else -1
        fan = 0
        if win >= 0:
            losers = sorted(s for i, s in enumerate(scores) if i != win)
            self_drawn = losers[0] == losers[-1]   # 自摸:三家失分相等
            fan = scores[win] // 3 - 8 if self_drawn else scores[win] - 24

        stats['games'] += 1
        stats['final_scores'].append(int(scores[train_seat]))
        stats['steps'].append(steps)
        if win == train_seat:
            stats['wins'] += 1
            stats['fans'].append(int(fan))
        if win == -1 and not illegal:
            stats['huang'] += 1

        if not traj:                               # 该局训练座位无多合法决策(极短局),跳过
            continue
        final_r = scores[train_seat] / reward_scale
        values = np.array([t[5] for t in traj], dtype=np.float32)
        T = len(traj)
        if reward_mode == 'terminal':
            adv, ret = compute_gae(values, final_r, gamma, lam)
        elif reward_mode == 'fanback':
            # 同一终局零和 R 沿轨迹从终局向前按 fanback_gamma 折扣、归一(Σw=1)再分摊
            kdist = np.arange(T - 1, -1, -1, dtype=np.float32)   # 距终局步数:末步=0
            w = fanback_gamma ** kdist
            w = w / w.sum()
            rewards = (final_r * w).astype(np.float32)
            adv, ret = compute_gae_rewards(values, rewards, gamma, lam)
            stats['shap_sum'] += float(rewards.sum())            # ≈ final_r(量级守恒自检)
            stats['shap_absmean'] += float(np.abs(rewards).mean())
            stats['shap_n'] += 1
        elif reward_mode == 'potential':
            vec_arr = np.stack([t[1] for t in traj]).astype(np.float32)
            phi = 1.0 - vec_arr[:, G_CUR_SHANTEN]                # (6−shanten)/6 ∈[0,1] 单调
            shap = np.zeros(T, dtype=np.float32)
            if T > 1:
                shap[:-1] = gamma * phi[1:] - phi[:-1]
            shap[-1] = gamma * 0.0 - phi[-1]                     # 终局势能=0 -> 保策略不变
            shap *= potential_coef
            rewards = shap.copy()
            rewards[-1] += final_r                               # 终局零和真值仍保留
            adv, ret = compute_gae_rewards(values, rewards, gamma, lam)
            stats['shap_sum'] += float(shap.sum())               # ≈ -c·Φ(s0)(telescope)
            stats['shap_absmean'] += float(np.abs(shap).mean())
            stats['shap_n'] += 1
        elif reward_mode == 'combo':
            # fanback 再分配 + potential 塑形 同时:R 只经 fanback 计一次(不再放终局),
            # potential 仅作叠加项。逐步奖励 = R·w(Σ=R 守恒) + c·(γΦ′−Φ)。
            kdist = np.arange(T - 1, -1, -1, dtype=np.float32)
            w = fanback_gamma ** kdist
            w = w / w.sum()
            fb = (final_r * w).astype(np.float32)                # 终局 R 时间再分配(Σ=R)
            vec_arr = np.stack([t[1] for t in traj]).astype(np.float32)
            phi = 1.0 - vec_arr[:, G_CUR_SHANTEN]
            shap = np.zeros(T, dtype=np.float32)
            if T > 1:
                shap[:-1] = gamma * phi[1:] - phi[:-1]
            shap[-1] = gamma * 0.0 - phi[-1]                      # 终局势能=0 -> 保策略不变
            shap *= potential_coef
            rewards = fb + shap                                   # 组合逐步奖励
            adv, ret = compute_gae_rewards(values, rewards, gamma, lam)
            stats['shap_sum'] += float(shap.sum())               # 仅记 potential 项量级
            stats['shap_absmean'] += float(np.abs(shap).mean())
            stats['shap_n'] += 1
        else:
            raise ValueError('unknown reward_mode %r' % reward_mode)
        trajs.append({
            'obs': np.stack([t[0] for t in traj]).astype(np.uint8),
            'vec': np.stack([t[1] for t in traj]).astype(np.float32),
            'mask': np.stack([t[2] for t in traj]).astype(np.uint8),
            'act': np.array([t[3] for t in traj], dtype=np.int64),
            'logp': np.array([t[4] for t in traj], dtype=np.float32),
            'adv': adv,
            'ret': ret,
        })
        stats['samples'] += len(traj)

    return trajs, stats
