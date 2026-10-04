"""micro-bench compute_vec_from_obs:真实 obs 分布上的 per-call 成本(前置2)。

出牌态(手牌%3==2)走 34 次换牌 shanten 的重路径,响应态走默认廉价路径。
shanten_useful_wait 带 lru_cache,故 warm 后大量命中。报 play 占比 + warm/cold us/call。
"""
import glob
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from feature_v2 import compute_vec_from_obs  # noqa: E402
import feature_v1 as fv1  # noqa: E402


def main():
    cands = (glob.glob('/root/autodl-tmp/mahjong/data_v1/obs*.npy')
             or glob.glob('/root/autodl-tmp/mahjong/data_v1/obs*'))
    obs = np.load(cands[0], mmap_mode='r')
    N = 20000
    random.seed(0)
    idx = sorted(random.sample(range(obs.shape[0]), N))
    sample = np.asarray(obs[idx])
    play = sum(1 for r in sample
               if int(r[fv1.CH_HAND:fv1.CH_HAND + 4].sum()) % 3 == 2)
    t0 = time.perf_counter()
    for r in sample:
        compute_vec_from_obs(r)
    t1 = time.perf_counter()
    for r in sample:
        compute_vec_from_obs(r)
    t2 = time.perf_counter()
    print('obs file: %s shape=%s' % (cands[0], obs.shape))
    print('N=%d  play-state(34-call) frac=%.1f%%' % (N, 100.0 * play / N))
    print('pass1 (cache warming): %.1f us/call  (%.0f calls/s)'
          % ((t1 - t0) / N * 1e6, N / (t1 - t0)))
    print('pass2 (cache warm):    %.1f us/call  (%.0f calls/s)'
          % ((t2 - t1) / N * 1e6, N / (t2 - t1)))


if __name__ == '__main__':
    main()
