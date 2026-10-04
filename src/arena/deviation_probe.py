"""Test #2(只读分析,不改评测主逻辑):argmax 同墙下"候选 vs 纯 M5"轨迹偏离统计。

问题:argmax + 1v3-vs-frozen-M5 下,RL 候选从 M5 微调而来(KL 锚),很多决策点的
argmax 与 M5 完全相同 → 整局与"纯 M5 在同墙"逐决策一致(零偏离)→ 名义 2000 局里
大量是 near-duplicate,有效独立样本远小于名义局数,配对差的真实方差被低估。

做法:对每个 group 复用 duplicate 的同一 (wall, wind),候选轮换四座;候选座位每个
决策点同时算 候选.act(argmax) 与 参照(M5).act(argmax),逐点比对是否相同。一局中只要
有一个"自由决策点(合法动作>1)"两者不同,即记该局"发生偏离"。报告:
  - 发生偏离的局占比;
  - 全体局配对差(score - others_mean,复现 run_arena 口径)与
    仅偏离局 / 仅零偏离局上的配对差 ± SE。
零偏离局的轨迹与纯 M5 完全一致,其对配对差的贡献在座位轮换下应抵消到 ~0;
信号全部来自偏离局——故"偏离局占比"直接标定有效样本量。

纯 CPU、argmax、只读:不修改 duplicate.py / run_arena.py 既有语义。
用法:
  python src/arena/deviation_probe.py \
    --candidate nnv2:checkpoints/rl_s3_potential/last.pt \
    --ref       nnv2:checkpoints/sl_v2_M5_gelu/best.pt \
    --groups 500 --seed 80640 --workers 24 --out results/arena_d/dev_s3pot.json
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

# 钉死 OMP(登录环境里 OMP_NUM_THREADS 可能为空 → libgomp 报错);单线程 CPU 推理足够
os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS') or '1'

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from arena.policies import make_policy  # noqa: E402
from arena.duplicate import make_wall  # noqa: E402
from feature_v1 import FeatureAgentV1  # noqa: E402
from env_fixed import MahjongGBEnvFixed as MahjongGBEnv  # noqa: E402
import random  # noqa: E402

_G = {}


def _init_worker(cand_spec, ref_spec, opp_spec, base_seed):
    _G['cand'] = make_policy(cand_spec)         # 候选(argmax)
    _G['ref'] = make_policy(ref_spec)           # 参照 M5(argmax),用于"M5 在此 obs 会怎么走"
    _G['opp'] = make_policy(opp_spec)           # 三个对手座位(默认= M5,argmax 可共享对象)
    _G['seed'] = base_seed


def _play_probe(env, cand, ref, opp, seat, wall, wind):
    """候选坐 seat、其余三座 opp;在 seat 的每个决策点比对 候选 vs 参照 的 argmax。"""
    obs = env.reset(prevalentWind=wind, tileWall=wall)
    done = False
    steps = 0
    n_free = 0      # 自由决策点(合法动作>1)
    n_diff = 0      # 候选 argmax != 参照 argmax 的自由决策点
    rewards = {}
    while not done:
        action_dict = {}
        for name, o in obs.items():
            s = int(name.split('_')[1]) - 1
            if s == seat:
                free = int(np.asarray(o['action_mask']).sum()) > 1
                a_c = cand.act(o)
                if free:
                    a_r = ref.act(o)
                    n_free += 1
                    if a_c != a_r:
                        n_diff += 1
                action_dict[name] = a_c
            else:
                action_dict[name] = opp.act(o)
        obs, rewards, done = env.step(action_dict)
        steps += 1
        assert steps < 1000, 'game not terminating'
    scores = [rewards['player_%d' % (i + 1)] for i in range(4)]
    return scores, n_free, n_diff


def _run_group(group_id):
    cand, ref, opp = _G['cand'], _G['ref'], _G['opp']
    base_seed = _G['seed']
    wall = make_wall((base_seed, group_id, 'wall'))
    wind = random.Random(str((base_seed, group_id, 'wind'))).randrange(4)
    env = MahjongGBEnv(config={'agent_clz': FeatureAgentV1})
    recs = []
    for seat in range(4):     # 候选轮换四座(复式)
        scores, n_free, n_diff = _play_probe(env, cand, ref, opp, seat, wall, wind)
        recs.append({
            'group': group_id, 'seat': seat,
            'score': scores[seat],
            'others_mean': (sum(scores) - scores[seat]) / 3.0,
            'n_free': n_free, 'n_diff': n_diff,
            'deviated': n_diff > 0,
        })
    return recs


def mean_se(xs):
    xs = np.asarray(xs, dtype=np.float64)
    if len(xs) == 0:
        return 0.0, 0.0, 0
    se = float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
    return float(xs.mean()), se, len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--ref', required=True, help='参照模型(M5),逐点比对 argmax 是否相同')
    ap.add_argument('--opponent', default=None, help='三对手座位,默认= --ref')
    ap.add_argument('--groups', type=int, default=500)
    ap.add_argument('--seed', type=int, default=80640)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    opp_spec = args.opponent or args.ref

    import time
    t0 = time.time()
    records = []
    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.candidate, args.ref, opp_spec, args.seed)) as pool:
        for recs in pool.imap_unordered(_run_group, range(args.groups), chunksize=4):
            records.extend(recs)
    elapsed = time.time() - t0

    n = len(records)
    dev_games = sum(1 for r in records if r['deviated'])
    dev_frac = dev_games / n if n else 0.0
    free_total = sum(r['n_free'] for r in records)
    diff_total = sum(r['n_diff'] for r in records)
    # 全体局配对差(复现 run_arena:按 group 先平均再总平均)
    by_group = defaultdict(list)
    for r in records:
        by_group[r['group']].append(r['score'] - r['others_mean'])
    d_all = mean_se([np.mean(v) for v in by_group.values()])
    # 仅偏离局 / 仅零偏离局(逐局口径)
    d_dev = mean_se([r['score'] - r['others_mean'] for r in records if r['deviated']])
    d_nodev = mean_se([r['score'] - r['others_mean'] for r in records if not r['deviated']])

    out = {
        'candidate': args.candidate, 'ref': args.ref, 'opponent': opp_spec,
        'groups': args.groups, 'seed': args.seed, 'cand_games': n, 'elapsed': round(elapsed, 1),
        'deviation_game_frac': round(dev_frac, 4),
        'deviation_games': dev_games, 'nondeviation_games': n - dev_games,
        'free_decisions_per_game': round(free_total / n, 2) if n else 0,
        'diff_decisions_per_game': round(diff_total / n, 3) if n else 0,
        'diff_rate_of_free': round(diff_total / free_total, 4) if free_total else 0,
        'paired_diff_all': {'mean': round(d_all[0], 3), 'se': round(d_all[1], 3), 'n_groups': d_all[2]},
        'paired_diff_deviation_only': {'mean': round(d_dev[0], 3), 'se': round(d_dev[1], 3), 'n': d_dev[2]},
        'paired_diff_nondev_only': {'mean': round(d_nodev[0], 3), 'se': round(d_nodev[1], 3), 'n': d_nodev[2]},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print('DEVIATION_PROBE_DONE')
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'summary': out, 'records': records}, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
