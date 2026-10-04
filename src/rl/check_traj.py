"""门禁2:轨迹一致性抽查。

把自对弈导出的牌谱(gate-smoke 时 worker0 记录的 logged_games_w0.pkl)喂
replay_driver + FeatureAgentV1 重建 obs,与 actor 当时记录的 (obs, mask) 逐元素比对。

用法:python src/rl/check_traj.py --pkl <run_dir>/logged_games_w0.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from feature_v1 import FeatureAgentV1  # noqa: E402
from replay_driver import replay_match  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    args = ap.parse_args()
    with open(args.pkl, 'rb') as f:
        games = pickle.load(f)

    n_cmp = 0
    for gi, (lines, traj) in enumerate(games):
        r = replay_match('selfplay_%d' % gi, lines, FeatureAgentV1)
        for seat, steps in traj.items():
            rebuilt = [(o['observation'], o['action_mask'])
                       for o in r['obs'][seat] if o['action_mask'].sum() > 1]
            assert len(rebuilt) == len(steps), \
                'game %d seat %d: %d rebuilt vs %d recorded' % (
                    gi, seat, len(rebuilt), len(steps))
            for t, ((ro, rm), s) in enumerate(zip(rebuilt, steps)):
                if not np.array_equal(ro, s[0]):
                    d = np.argwhere(ro != s[0])
                    raise AssertionError('game %d seat %d step %d obs differs ch %s'
                                         % (gi, seat, t, np.unique(d[:, 0])[:8]))
                if not np.array_equal(rm, s[1]):
                    raise AssertionError('game %d seat %d step %d mask differs' % (gi, seat, t))
                n_cmp += 1
    print('GATE_TRAJ_PASS: %d games, %d decisions compared elementwise' % (len(games), n_cmp))


if __name__ == '__main__':
    main()
