# -*- coding: utf-8 -*-
"""Gen2 预研 gate:完美信息(oracle)价值头离线可行性实验。

从 data.txt 重放 N 局,在每个决策点(合法动作>1)提取:
  - 现有 102 通道不完美信息 obs(feature_v1,与线上完全一致)
  - 28 通道 oracle 附加:三家暗手牌温度计 4ch×3、牌墙剩余构成温度计 4ch、
    四家(自/下/对/上)向听距离温度计 sh<=0/<=1/<=2 共 3ch×4
目标 = 本座位终局得分 / 30(与 M2/M3 价值口径一致)。
同一小型 CNN、同一训练预算,分别用 102ch(基线)与 130ch(oracle)训练,
报告各自 held-out(按局切分)Pearson。gate:oracle > 0.55(基线参照 ~0.30)。

CPU 即可:python scripts/oracle_gate.py --games 6000 --workers 12
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

# M1 教训:shanten lru_cache 无帽 + 多进程在小内存(无卡实例 2GB)上会 OOM 活锁。
# 必须在 import feature_v1 前钉死缓存帽(feature_v1 在 import 时读这些 env)。
for _k in ('SHANTEN_CACHE_WAIT', 'SHANTEN_CACHE_DISCARD',
           'SHANTEN_CACHE_L1', 'SHANTEN_CACHE_L2'):
    os.environ.setdefault(_k, '30000')
os.environ.setdefault('OMP_NUM_THREADS', '1')

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'src')))
from feature_v1 import FeatureAgentV1, shanten_useful_wait, \
    shanten_useful_discard  # noqa: E402
from replay_driver import iter_matches, replay_match  # noqa: E402

TILE_LIST = FeatureAgentV1.TILE_LIST
TIDX = {t: i for i, t in enumerate(TILE_LIST)}
ORACLE_CH = 28


def tiles_to_thermo(counts, planes, base):
    """counts: dict tile->n;写入 planes[base:base+4] 的 (4,9) 温度计。"""
    for tile, n in counts.items():
        i = TIDX[tile]
        r, c = i // 9, i % 9
        for k in range(min(n, 4)):
            planes[base + k][r][c] = 1


def hand_shanten(hand):
    h = tuple(sorted(hand))
    try:
        if len(h) % 3 == 1:
            return shanten_useful_wait(h)[0]
        if len(h) % 3 == 2:
            return shanten_useful_discard(h)[0]
    except Exception:
        pass
    return 6


class OracleObserver(object):
    """挂在 replay_match 上:维护墙余量账本,在决策点快照 oracle 通道。"""

    def __init__(self):
        self.wall = {}
        self.lens = [0, 0, 0, 0]
        self.out = [[] for _ in range(4)]   # 每座位 [(step_idx, oracle28), ...]

    def __call__(self, t, agents, obs):
        if t[0] == 'Wind':
            self.wall = {tile: 4 for tile in TILE_LIST}
            self.lens = [0, 0, 0, 0]
            return
        if t[0] == 'Player':
            if t[2] == 'Deal':
                for tile in t[3:]:
                    self.wall[tile] -= 1
            elif t[2] == 'Draw' and len(t) > 3:
                self.wall[t[3]] -= 1
        grown = [s for s in range(4) if len(obs[s]) > self.lens[s]]
        if not grown:
            return
        sh = None
        for s in grown:
            self.lens[s] = len(obs[s])
            o = obs[s][-1]
            if o['action_mask'].sum() <= 1:
                continue
            if sh is None:   # 向听对本行所有 grown 座位共享计算一次
                sh = [hand_shanten(agents[j].hand) for j in range(4)]
            pl = np.zeros((ORACLE_CH, 4, 9), dtype=np.uint8)
            for rel in (1, 2, 3):     # 下/对/上 家暗手牌
                j = (s + rel) % 4
                cnt = {}
                for tile in agents[j].hand:
                    cnt[tile] = cnt.get(tile, 0) + 1
                tiles_to_thermo(cnt, pl, (rel - 1) * 4)
            tiles_to_thermo({k: v for k, v in self.wall.items() if v > 0},
                            pl, 12)   # 牌墙剩余构成
            for rel in range(4):      # 自/下/对/上 向听温度计
                v = sh[(s + rel) % 4]
                for k in range(3):
                    if v <= k:
                        pl[16 + rel * 3 + k][:][:] = 1
            self.out[s].append((self.lens[s] - 1, pl))


def extract_chunk(args_tuple):
    path, skip, count = args_tuple
    X102, X28, Y = [], [], []
    n = 0
    for mid, lines in iter_matches(path, limit=skip + count):
        n += 1
        if n <= skip:
            continue
        ob = OracleObserver()
        r = replay_match(mid, lines, FeatureAgentV1, observer=ob)
        for s in range(4):
            for step, pl in ob.out[s]:
                X102.append(r['obs'][s][step]['observation'])
                X28.append(pl)
                Y.append(r['scores'][s])
    return (np.stack(X102).astype(np.uint8), np.stack(X28),
            np.array(Y, dtype=np.float32))


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def train_eval(in_ch, Xtr, Ytr, Xva, Yva, epochs, seed):
    import torch
    from torch import nn
    from dataset_v1 import normalize_obs
    torch.manual_seed(seed)
    # 钉 1 线程:无卡 0.5 核实例上 set_num_threads(24) 会让 conv 的 intra-op
    # 在半核上疯狂上下文切换,训练慢几十倍(实测训练阶段久不出 epoch)。
    # 小网单线程足够,且 os.cpu_count() 不反映 cgroup 带宽限制,不能依赖。
    torch.set_num_threads(1)

    class Block(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.n1, self.c1 = nn.GroupNorm(8, ch), nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
            self.n2, self.c2 = nn.GroupNorm(8, ch), nn.Conv2d(ch, ch, 3, 1, 1, bias=False)

        def forward(self, x):
            y = self.c1(torch.relu(self.n1(x)))
            return x + self.c2(torch.relu(self.n2(y)))

    net = nn.Sequential(
        nn.Conv2d(in_ch, 48, 3, 1, 1, bias=False),
        Block(48), Block(48), Block(48), Block(48),
        nn.GroupNorm(8, 48), nn.ReLU(),
        nn.Conv2d(48, 16, 1), nn.Flatten(),
        nn.Linear(16 * 36, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def norm(x_np):
        # 前 102 通道按线上口径 normalize,oracle 附加通道本就是 0/1
        x = torch.from_numpy(x_np)
        o = normalize_obs(x[:, :102])
        if x.shape[1] > 102:
            o = torch.cat([o, x[:, 102:].float()], dim=1)
        return o

    ntr = len(Ytr)
    ytr = torch.from_numpy(Ytr / 30.0)
    for ep in range(epochs):
        perm = np.random.RandomState(seed + ep).permutation(ntr)
        tot = 0.0
        for s in range(0, ntr, 512):
            bidx = perm[s:s + 512]
            pred = net(norm(Xtr[bidx][:, :in_ch])).squeeze(-1)
            loss = nn.functional.smooth_l1_loss(pred, ytr[bidx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(bidx)
        print('  [in=%d] epoch %d loss %.4f' % (in_ch, ep + 1, tot / ntr), flush=True)
    preds = []
    with torch.no_grad():
        for s in range(0, len(Yva), 1024):
            preds.append(net(norm(Xva[s:s + 1024][:, :in_ch])).squeeze(-1).numpy())
    return pearson(np.concatenate(preds), Yva / 30.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/root/autodl-tmp/mahjong/baseline/SL/data/data.txt')
    ap.add_argument('--games', type=int, default=6000)
    ap.add_argument('--val-games', type=int, default=600)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20260613)
    args = ap.parse_args()

    t0 = time.time()
    chunk = max(args.games // args.workers, 1)
    tasks = [(args.data, i * chunk, min(chunk, args.games - i * chunk))
             for i in range(args.workers) if i * chunk < args.games]
    if args.workers == 1:
        # 单进程内联提取:避免 Pool 把大数组 pickle 回父进程的 2× 峰值,
        # 也避免 worker 被 OOM-kill 后 pool.map 死等(2GB 实例上的真实故障)
        parts = [extract_chunk(tasks[0])]
    else:
        with Pool(args.workers) as pool:
            parts = pool.map(extract_chunk, tasks)
    X102 = np.concatenate([p[0] for p in parts])
    X28 = np.concatenate([p[1] for p in parts])
    Y = np.concatenate([p[2] for p in parts])
    X = np.concatenate([X102, X28], axis=1)
    del X102, X28, parts
    # 末尾 val_games 局近似按样本比例切(局间无泄漏:extract 按局序排列)
    n_val = int(len(Y) * (args.val_games / float(args.games)))
    Xtr, Ytr, Xva, Yva = X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]
    print('extracted %d samples (train %d / val %d) in %.1fs'
          % (len(Y), len(Ytr), len(Yva), time.time() - t0), flush=True)

    r_base = train_eval(102, Xtr, Ytr, Xva, Yva, args.epochs, args.seed)
    print('baseline(102ch) val Pearson = %.4f' % r_base, flush=True)
    r_orac = train_eval(102 + ORACLE_CH, Xtr, Ytr, Xva, Yva, args.epochs, args.seed)
    print('oracle(130ch)  val Pearson = %.4f' % r_orac, flush=True)
    print(json.dumps({'n_train': len(Ytr), 'n_val': len(Yva),
                      'pearson_base102': round(r_base, 4),
                      'pearson_oracle130': round(r_orac, 4),
                      'gate(>0.55)': r_orac > 0.55}))
    print('ORACLE_GATE_DONE')


if __name__ == '__main__':
    main()
