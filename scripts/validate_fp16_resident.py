# -*- coding: utf-8 -*-
"""fp16 常驻推理 vs 现行 fp32 部署路径:真实决策点 argmax 一致率 + 单样本延迟。

- 决策点取 data_v1 的 obs/mask memmap,只取合法动作数 >1 的行(单合法行恒一致);
- 两路径输入完全相同(同一 normalize),fp32 路径即现行 bot 的加载方式;
- 不一致样本记录两路 top1/top2 logit 差,核对是否打平翻转。
严格 py3.6 语法,在 botzone-compat 环境运行。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'src')))

import torch  # noqa: E402
torch.set_num_threads(1)
from model_v1 import MahjongNetV1, FP16ResidentNet  # noqa: E402
from feature_v1 import SCALED_CHANNELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data', default='/root/autodl-tmp/mahjong/data_v1')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20260612)
    args = ap.parse_args()

    # .dat 由 open_memmap 生成,带 npy 头,必须 np.load 读(裸 memmap 错位 128B)
    obs_mm = np.load(os.path.join(args.data, 'obs.dat'), mmap_mode='r')
    mask_mm = np.load(os.path.join(args.data, 'mask.dat'), mmap_mode='r')
    rng = np.random.RandomState(args.seed)
    idx = []
    while len(idx) < args.n:
        for i in rng.randint(0, len(obs_mm), size=args.n):
            if int(mask_mm[i].sum()) > 1:
                idx.append(int(i))
                if len(idx) >= args.n:
                    break

    scale = torch.ones(102, 1, 1)
    for c in SCALED_CHANNELS:
        scale[c] = 1.0 / 255

    payload = torch.load(args.weights, map_location='cpu')
    net16 = FP16ResidentNet(payload['state_dict'], payload['config'])
    net32, _ = MahjongNetV1.load(args.weights)
    net32.eval()

    agree = 0
    mismatches = []
    t16, t32 = [], []
    with torch.no_grad():
        for i in idx:
            o = torch.from_numpy(np.ascontiguousarray(obs_mm[i])).float().unsqueeze(0)
            o = o * scale
            m = torch.from_numpy(np.ascontiguousarray(mask_mm[i])).float().unsqueeze(0)
            neg = (1.0 - m) * (-1e9)
            t0 = time.perf_counter()
            l16, _ = net16(o)
            t16.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            l32, _ = net32(o)
            t32.append((time.perf_counter() - t0) * 1000)
            l16 = l16 + neg
            l32 = l32 + neg
            a16 = int(l16.argmax(1).item())
            a32 = int(l32.argmax(1).item())
            if a16 == a32:
                agree += 1
            else:
                v16 = torch.topk(l16[0], 2).values
                v32 = torch.topk(l32[0], 2).values
                mismatches.append({
                    'idx': i, 'a16': a16, 'a32': a32,
                    'gap16_top1_top2': round(float(v16[0] - v16[1]), 6),
                    'gap32_top1_top2': round(float(v32[0] - v32[1]), 6),
                    'logit_delta': round(float((l16[0][a32] - l16[0][a16]).abs()), 6),
                })
    t16, t32 = np.array(t16), np.array(t32)
    print(json.dumps({
        'n': len(idx), 'agree': agree, 'agree_rate': round(agree / float(len(idx)), 5),
        'mismatches': mismatches[:20],
        'lat_ms_fp16_resident': {'mean': round(float(t16.mean()), 2),
                                 'p50': round(float(np.percentile(t16, 50)), 2),
                                 'p99': round(float(np.percentile(t16, 99)), 2)},
        'lat_ms_fp32': {'mean': round(float(t32.mean()), 2),
                        'p50': round(float(np.percentile(t32, 50)), 2),
                        'p99': round(float(np.percentile(t32, 99)), 2)},
    }, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
