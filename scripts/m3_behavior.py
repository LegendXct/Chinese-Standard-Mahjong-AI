# -*- coding: utf-8 -*-
"""行为对比:同一批真实决策点上,人类标签 / SL champion / RL last 的动作类型分布。

20000 个 val 决策点(合法动作>1),batched CPU argmax。输出 JSON 表。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'src')))
import torch  # noqa: E402
from model_v1 import MahjongNetV1, InputDictAdapter  # noqa: E402
from dataset_v1 import normalize_obs  # noqa: E402

RANGES = [('Pass', 0, 1), ('Hu', 1, 2), ('Play', 2, 36), ('Chi', 36, 99),
          ('Peng', 99, 133), ('Gang', 133, 167), ('AnGang', 167, 201),
          ('BuGang', 201, 235)]


def type_dist(acts):
    n = float(len(acts))
    return {name: round(int(((acts >= lo) & (acts < hi)).sum()) / n, 4)
            for name, lo, hi in RANGES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/root/autodl-tmp/mahjong/data_v1')
    ap.add_argument('--sl', default='checkpoints/sl_B_gn/best.pt')
    ap.add_argument('--rl', default='checkpoints/rl_v1/last.pt')
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=20260613)
    args = ap.parse_args()
    torch.set_num_threads(24)

    # 注意:这些 .dat 由 np.lib.format.open_memmap 生成,带 128 字节 npy 头,
    # 必须 np.load(mmap_mode) 读;裸 np.memmap 会整体错位 128 字节(已踩坑)
    obs = np.load(os.path.join(args.data, 'obs.dat'), mmap_mode='r')
    mask = np.load(os.path.join(args.data, 'mask.dat'), mmap_mode='r')
    act = np.load(os.path.join(args.data, 'act.dat'), mmap_mode='r')
    n_all = len(obs)
    rng = np.random.RandomState(args.seed)
    idx = []
    while len(idx) < args.n:
        for i in rng.randint(0, n_all, size=args.n):
            if int(mask[i].sum()) > 1:
                idx.append(int(i))
                if len(idx) >= args.n:
                    break
    idx = np.array(sorted(idx))

    nets = {}
    for tag, path in (('SL', args.sl), ('RL', args.rl)):
        net, _ = MahjongNetV1.load(path)
        net.eval()
        nets[tag] = net

    preds = {t: [] for t in nets}
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            sl = idx[s:s + 512]
            o = normalize_obs(torch.from_numpy(np.ascontiguousarray(obs[sl])))
            m = torch.from_numpy(np.ascontiguousarray(mask[sl])).float()
            neg = (1.0 - m) * (-1e9)
            for t, net in nets.items():
                logits, _ = net(o)
                preds[t].append((logits + neg).argmax(1).numpy())
    human = act[idx].astype(np.int64)
    out = {'n': len(idx), 'dist': {'human': type_dist(human)}}
    for t in nets:
        p = np.concatenate(preds[t])
        out['dist'][t] = type_dist(p)
        out['acc_vs_human_%s' % t] = round(float((p == human).mean()), 4)
    p_sl, p_rl = np.concatenate(preds['SL']), np.concatenate(preds['RL'])
    out['sl_rl_agree'] = round(float((p_sl == p_rl).mean()), 4)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
