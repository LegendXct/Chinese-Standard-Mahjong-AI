# -*- coding: utf-8 -*-
"""Path A CPU R² gate:测"隐藏运气方差"是否存在(决定 RL 是否还值得开 GPU)。

判别 RL 在 M5 上 parity 的两解释:
  解释1(信噪比):运气方差淹没梯度 → 运气基线可修 → Path A 值得开一次 GPU。
  解释2(真天花板):M5 argmax 对当前架构+特征已近最优 → 任何 RL 修不了 → 收口转架构。

方法:同一批"逐决策点 → 该座位终局 score/30"样本上,比较【同架构同容量】两个预测器的解释力:
  - obs-only:输入 = 归一化 obs(102ch) + vec(107),oracle 28ch 喂 0;
  - perfect-info:输入 = 同上 + oracle 28ch(三家当前暗手牌温度计 + 牌墙剩余构成 + 四家向听)。
两者【架构/参数/训练完全相同,唯一差异是 oracle 通道是真值还是 0】→ R² 差纯来自信息、非容量。
增量 R²(perfect − obs)= obs 价值头看不到、完美信息能解释的那块运气方差 = 运气基线能扣掉的量。

注:oracle 用【决策点当前】隐藏态(对手当前手牌 + 牌墙剩余),是该点的充分统计量,
≥ 初始发牌信息 → 增量是"任何运气基线(含初始发牌型)可扣方差"的【上界】。若上界都小,
则初始发牌型必更小,解释2 稳。

参照(防 obs-only 小网低估 obs):额外直接前向 M5 与 rl_s3_potential 的【价值头】在同一 val 上的
R²。obs 上限 = max(obs-only 预测器, 各价值头)。

口径:终局零和 score/30;按【局】切分 train/val(同局不跨集);R² := Pearson r 的平方
(最优线性解释方差,scale/bias 无关,对训练预测器与价值头都公平可比)。
纯 CPU,不训练 PPO。用法:
  python scripts/path_a_gate.py --games 8000 --val-games 800 --workers 24 --epochs 6
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS') or '1'
for _k in ('SHANTEN_CACHE_WAIT', 'SHANTEN_CACHE_DISCARD', 'SHANTEN_CACHE_L1', 'SHANTEN_CACHE_L2'):
    os.environ.setdefault(_k, '30000')

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, '..', 'src')))
sys.path.insert(0, _HERE)   # for oracle_gate

from feature_v1 import FeatureAgentV1  # noqa: E402
from feature_v2 import compute_vec_from_obs, VEC_DIM  # noqa: E402
from replay_driver import iter_matches, replay_match  # noqa: E402
from oracle_gate import OracleObserver, ORACLE_CH  # noqa: E402  (复用 oracle 通道重建)


def extract_chunk(args_tuple):
    path, skip, count = args_tuple
    X, XV, ORC, Y, GID = [], [], [], [], []
    n = 0
    for mid, lines in iter_matches(path, limit=skip + count):
        n += 1
        if n <= skip:
            continue
        gid = n - 1                      # 全局 0-based 局号(各 chunk 不重叠)
        ob = OracleObserver()
        r = replay_match(mid, lines, FeatureAgentV1, observer=ob)
        for s in range(4):
            for step, pl in ob.out[s]:   # 仅 mask>1 的自由决策点
                obs = r['obs'][s][step]['observation']
                X.append(obs.astype(np.uint8))
                XV.append(compute_vec_from_obs(obs).astype(np.float32))
                ORC.append(pl.astype(np.uint8))
                Y.append(r['scores'][s])
                GID.append(gid)
    if not Y:
        return None
    return (np.stack(X), np.stack(XV), np.stack(ORC),
            np.array(Y, np.float32), np.array(GID, np.int32))


def r2_of(pred, target):
    """R² := Pearson r 的平方(最优线性解释方差,scale/bias 无关)。返回 (r, r2)。"""
    pred = np.asarray(pred, np.float64); target = np.asarray(target, np.float64)
    if pred.std() < 1e-9 or target.std() < 1e-9:
        return 0.0, 0.0
    r = float(np.corrcoef(pred, target)[0, 1])
    return r, r * r


def build_net(seed):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    IN = 102 + ORACLE_CH      # 130:obs-only 把后 28ch 喂 0,perfect 喂真值

    class Block(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.n1 = nn.GroupNorm(8, ch); self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
            self.n2 = nn.GroupNorm(8, ch); self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)

        def forward(self, x):
            y = self.c1(torch.nn.functional.gelu(self.n1(x)))
            return x + self.c2(torch.nn.functional.gelu(self.n2(y)))

    class ValNet(nn.Module):
        def __init__(self, w=128, blocks=5):
            super().__init__()
            self.stem = nn.Conv2d(IN, w, 3, 1, 1, bias=False)
            self.blocks = nn.Sequential(*[Block(w) for _ in range(blocks)])
            self.gn = nn.GroupNorm(8, w)
            self.proj = nn.Conv2d(w, 32, 1)
            self.mlp = nn.Sequential(
                nn.Linear(32 * 36 + VEC_DIM, 512), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1))

        def forward(self, obs, vec):
            x = self.blocks(self.stem(obs))
            x = self.proj(torch.nn.functional.gelu(self.gn(x)))
            x = x.flatten(1)
            return self.mlp(torch.cat([x, vec], dim=1)).squeeze(-1)

    return ValNet()


def train_predictor(use_oracle, X102, ORC, VEC, Y, tr, va, epochs, seed, bs=1024):
    """同架构同 seed;use_oracle=False 时把 oracle 28ch 喂 0(唯一差异)。返回 val (r, r2)。"""
    import torch
    from torch import nn
    from dataset_v2 import normalize_obs
    torch.manual_seed(seed)
    torch.set_num_threads(16)
    net = build_net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ytr = torch.from_numpy(Y[tr] / 30.0)

    def make_in(idx):
        o = normalize_obs(torch.from_numpy(X102[idx]))           # (B,102,4,9) float
        if use_oracle:
            orc = torch.from_numpy(ORC[idx]).float()
        else:
            orc = torch.zeros((len(idx), ORACLE_CH, 4, 9), dtype=torch.float32)
        obs = torch.cat([o, orc], dim=1)
        return obs, torch.from_numpy(VEC[idx])

    ntr = len(tr)
    best = (0.0, -1.0, 0)   # (r, r2, epoch) at peak val —— 早停口径,避开后期过拟合
    for ep in range(epochs):
        net.train()
        perm = tr[np.random.RandomState(seed + ep).permutation(ntr)]
        tot = 0.0
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            obs, vec = make_in(idx)
            pred = net(obs, vec)
            loss = nn.functional.smooth_l1_loss(pred, torch.from_numpy(Y[idx] / 30.0))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        net.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(va), 2048):
                idx = va[i:i + 2048]
                obs, vec = make_in(idx)
                preds.append(net(obs, vec).numpy())
        r, r2 = r2_of(np.concatenate(preds), Y[va] / 30.0)
        if r2 > best[1]:
            best = (r, r2, ep + 1)
        print('  [oracle=%d] ep %2d trainloss %.4f  val r=%.4f R2=%.4f  (peak R2=%.4f @ep%d)'
              % (use_oracle, ep + 1, tot / ntr, r, r2, best[1], best[2]), flush=True)
    return best


def eval_value_head(ckpt, X102, VEC, Y, va):
    """直接前向 MahjongNetV2 价值头(obs102+vec),返回 val (r, r2)。"""
    import torch
    from model_v2 import MahjongNetV2
    from dataset_v2 import normalize_obs
    torch.set_num_threads(16)
    try:
        net, _ = MahjongNetV2.load(ckpt, map_location='cpu')
    except Exception as e:
        print('  value-head load fail %s: %r' % (ckpt, e), flush=True)
        return None
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(va), 2048):
            idx = va[i:i + 2048]
            o = normalize_obs(torch.from_numpy(X102[idx]))
            v = torch.from_numpy(VEC[idx])
            _, val = net(o, v)
            preds.append(np.asarray(val).reshape(-1))
    return r2_of(np.concatenate(preds), Y[va] / 30.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/root/autodl-tmp/mahjong/baseline/SL/data/data.txt')
    ap.add_argument('--games', type=int, default=8000)
    ap.add_argument('--val-games', type=int, default=800)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20260630)
    ap.add_argument('--m5', default='/root/autodl-tmp/mahjong/checkpoints/sl_v2_M5_gelu/best.pt')
    ap.add_argument('--rl', default='/root/autodl-tmp/mahjong/checkpoints/rl_s3_potential/last.pt')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    t0 = time.time()
    chunk = max(args.games // args.workers, 1)
    tasks = [(args.data, i * chunk, min(chunk, args.games - i * chunk))
             for i in range(args.workers) if i * chunk < args.games]
    with Pool(args.workers) as pool:
        parts = [p for p in pool.map(extract_chunk, tasks) if p is not None]
    X102 = np.concatenate([p[0] for p in parts])
    VEC = np.concatenate([p[1] for p in parts])
    ORC = np.concatenate([p[2] for p in parts])
    Y = np.concatenate([p[3] for p in parts])
    GID = np.concatenate([p[4] for p in parts])
    del parts
    val_mask = GID >= (args.games - args.val_games)
    va = np.flatnonzero(val_mask); tr = np.flatnonzero(~val_mask)
    print('extracted %d samples (train %d / val %d) over %d games in %.1fs'
          % (len(Y), len(tr), len(va), args.games, time.time() - t0), flush=True)
    print('val target score/30: mean %.3f std %.3f' % (Y[va].mean() / 30, Y[va].std() / 30), flush=True)

    r_obs, r2_obs, ep_obs = train_predictor(False, X102, ORC, VEC, Y, tr, va, args.epochs, args.seed)
    print('OBS-ONLY  predictor: peak r=%.4f  R2=%.4f @ep%d' % (r_obs, r2_obs, ep_obs), flush=True)
    r_prf, r2_prf, ep_prf = train_predictor(True, X102, ORC, VEC, Y, tr, va, args.epochs, args.seed)
    print('PERFECT   predictor: peak r=%.4f  R2=%.4f @ep%d' % (r_prf, r2_prf, ep_prf), flush=True)

    vh = {}
    for name, ck in (('M5', args.m5), ('rl_s3pot', args.rl)):
        res = eval_value_head(ck, X102, VEC, Y, va)
        if res:
            vh[name] = res
            print('VALUE-HEAD %-8s: r=%.4f  R2=%.4f' % (name, res[0], res[1]), flush=True)

    # 主判据 = 控表征增量(同架构同训练,唯一差异是 oracle 信息):隔离信息价值、不受表征质量影响
    incr_ctrl = r2_prf - r2_obs
    obs_ceiling = max([r2_obs] + [v[1] for v in vh.values()])
    incr_cons = r2_prf - obs_ceiling   # 保守:perfect 超过 obs 上界(含 M5 价值头)
    incr = incr_ctrl
    verdict = ('解释1(信噪比)→ Path A 值得开一次 GPU' if incr >= 0.15
               else '解释2(真天花板)→ 收口 RL,转架构/特征/数据' if incr < 0.05
               else '中间档 → 需自对弈数据复核')
    out = {
        'games': args.games, 'val_games': args.val_games, 'n_train': int(len(tr)), 'n_val': int(len(va)),
        'epochs': args.epochs,
        'r2_obs_only': round(r2_obs, 4), 'r2_perfect': round(r2_prf, 4),
        'r2_value_heads': {k: round(v[1], 4) for k, v in vh.items()},
        'obs_ceiling_r2': round(obs_ceiling, 4),
        'increment_r2_controlled(perfect-obs_only)': round(incr_ctrl, 4),
        'increment_r2_conservative(perfect-ceiling)': round(incr_cons, 4),
        'verdict': verdict,
    }
    print('==== GATE RESULT ====', flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    print('PATH_A_GATE_DONE', flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(out, open(args.out, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
