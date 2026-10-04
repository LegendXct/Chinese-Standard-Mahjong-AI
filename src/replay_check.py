"""全量重放合法性校验:对每个决策点断言"数据记录的动作 ∈ 生成的 action_mask"。

用法(服务器项目根目录,tmux 中):
  python src/replay_check.py --data baseline/SL/data/data.txt --workers 128          # 全量
  python src/replay_check.py --data baseline/SL/data/sample.txt --workers 4          # 冒烟
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_v1 import FeatureAgentV1  # noqa: E402
from replay_driver import replay_match, check_legality  # noqa: E402
from preprocess_v1 import index_match_offsets, _iter_matches_in_slice  # noqa: E402


def check_chunk(args):
    chunk_id, path, start, end, first_gid = args
    violations = []
    n_games = n_decisions = 0
    gid = first_gid
    for match_id, lines in _iter_matches_in_slice(path, start, end):
        r = replay_match(match_id, lines, FeatureAgentV1)
        n_decisions += sum(len(x) for x in r['act'])
        for v in check_legality(r):
            violations.append({'game': gid, 'match_id': match_id,
                               'seat': v[0], 'step': v[1], 'act': v[2], 'mask_sum': v[3]})
        n_games += 1
        gid += 1
    return chunk_id, n_games, n_decisions, violations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--workers', type=int, default=128)
    ap.add_argument('--out', default=None, help='违规明细输出 json 路径')
    args = ap.parse_args()

    t0 = time.time()
    offsets, fsize = index_match_offsets(args.data)
    n_games = len(offsets)
    n_chunks = min(args.workers * 2, n_games)
    bnd = np.linspace(0, n_games, n_chunks + 1).astype(int)
    tasks = []
    for c in range(n_chunks):
        lo, hi = bnd[c], bnd[c + 1]
        if lo == hi:
            continue
        tasks.append((c, args.data, offsets[lo], offsets[hi] if hi < n_games else fsize, lo))

    tot_games = tot_dec = 0
    violations = []
    with Pool(args.workers) as pool:
        for cid, g, d, v in pool.imap_unordered(check_chunk, tasks):
            tot_games += g
            tot_dec += d
            violations.extend(v)
    elapsed = time.time() - t0
    print('games: %d, decisions: %d, violations: %d, %.1fs (workers=%d)' % (
        tot_games, tot_dec, len(violations), elapsed, args.workers))
    if violations:
        for v in violations[:20]:
            print(v)
        if args.out:
            with open(args.out, 'w') as f:
                json.dump(violations, f, indent=2)
            print('full list -> %s' % args.out)
    sys.exit(1 if violations else 0)


if __name__ == '__main__':
    main()
