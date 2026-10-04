"""从 v1 的 obs.dat 现算 vec.dat(feature_v2.compute_vec_from_obs),与 v1 样本逐行对齐。

不重放对局——vec 是 obs 的确定性函数,直接读 obs.dat 第 i 行算第 i 行 vec,
天然对齐 v1 的 obs/mask/act/split/meta,且只新增 ~1.3GB(float16)磁盘。

用法:python src/build_vec_v2.py --data /root/autodl-tmp/mahjong/data_v1 --workers 24
"""
import argparse
import json
import os
import sys
import time

# M1 教训:多进程 + 无帽 shanten 缓存是内存放大器;导入 feature 前钉帽
for _k in ('SHANTEN_CACHE_WAIT', 'SHANTEN_CACHE_DISCARD',
           'SHANTEN_CACHE_L1', 'SHANTEN_CACHE_L2'):
    os.environ.setdefault(_k, '262144')

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_v2 import compute_vec_from_obs, VEC_DIM  # noqa: E402

_G = {}


def _init(obs_path, vec_path, n, ch):
    _G['obs'] = np.load(obs_path, mmap_mode='r')
    _G['vec'] = np.lib.format.open_memmap(vec_path, mode='r+')


def _work(rng):
    s, e = rng
    obs = _G['obs']
    vec = _G['vec']
    for i in range(s, e):
        vec[i] = compute_vec_from_obs(np.asarray(obs[i])).astype(np.float16)
    return e - s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/root/autodl-tmp/mahjong/data_v1')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--chunk', type=int, default=20000)
    args = ap.parse_args()

    obs_path = os.path.join(args.data, 'obs.dat')
    vec_path = os.path.join(args.data, 'vec.dat')
    obs = np.load(obs_path, mmap_mode='r')
    n = len(obs)
    ch = obs.shape[1]
    del obs
    # 预分配 vec.dat
    np.lib.format.open_memmap(vec_path, mode='w+', dtype=np.float16,
                              shape=(n, VEC_DIM))
    print('building vec.dat: n=%d dim=%d -> %.2f GB' % (
        n, VEC_DIM, n * VEC_DIM * 2 / 2**30), flush=True)

    ranges = [(s, min(s + args.chunk, n)) for s in range(0, n, args.chunk)]
    t0 = time.time()
    from multiprocessing import Pool
    done = 0
    with Pool(args.workers, initializer=_init,
              initargs=(obs_path, vec_path, n, ch)) as pool:
        for k in pool.imap_unordered(_work, ranges):
            done += k
            if done % 500000 < args.chunk:
                el = time.time() - t0
                print('  %d/%d (%.0f%%) %.0fs eta %.0fs' % (
                    done, n, 100.0 * done / n, el, el * (n - done) / max(done, 1)),
                    flush=True)
    with open(os.path.join(args.data, 'vec_info.json'), 'w') as f:
        json.dump({'vec_dim': VEC_DIM, 'n': n, 'feature_version': 'v2',
                   'dtype': 'float16'}, f)
    print('vec.dat done in %.0fs (%.1f min)' % (
        time.time() - t0, (time.time() - t0) / 60), flush=True)
    print('BUILD_VEC_DONE')


if __name__ == '__main__':
    main()
