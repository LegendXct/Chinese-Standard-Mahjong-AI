"""复式评测场 CLI。

用法(服务器项目根目录):
  python src/arena/run_arena.py --candidate rule --opponent random --groups 500 --workers 64 \
      --out results/arena_rule_vs_random.md
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from arena.policies import make_policy  # noqa: E402
from arena.duplicate import play_group  # noqa: E402

_G = {}


def _init_worker(cand_spec, opp_spec, base_seed, mode, temperature, opponents,
                 indep_opp, decision_salt):
    def withT(spec):
        # 对称温度:>0 时给 NN spec 追加 @tT(已带 @t 的不动);rule/random 无温度
        if temperature and temperature > 0 and spec.startswith('nn') and '@t' not in spec:
            return spec + '@t%g' % temperature
        return spec
    _G['cand'] = make_policy(withT(cand_spec))
    _G['seed'] = base_seed
    _G['mode'] = mode
    _G['salt'] = decision_salt
    _G['opp_list'] = None
    if opponents:                       # 异质三对手(Test#3a)
        specs = [s.strip() for s in opponents.split(',')]
        assert len(specs) == 3, '--opponents 需 3 个逗号分隔 spec(1v3)'
        _G['opp_list'] = [make_policy(withT(s)) for s in specs]
        _G['opp'] = _G['opp_list'][0]   # 占位(play_group 用 opp_list)
    else:
        opp = make_policy(withT(opp_spec))
        _G['opp'] = opp
        if indep_opp:                   # 同质对手但每座独立采样 rng(Test#1 T>0)
            _G['opp_list'] = [opp, opp.clone_shared(), opp.clone_shared()]


def _run_group(group_id):
    return play_group(group_id, _G['seed'], _G['cand'], _G['opp'], _G['mode'],
                      opp_policies=_G['opp_list'], decision_salt=_G['salt'])


def mean_se(xs):
    xs = np.asarray(xs, dtype=np.float64)
    if len(xs) == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', default='rule')
    ap.add_argument('--opponent', default='random')
    ap.add_argument('--groups', type=int, default=500)
    ap.add_argument('--mode', default='1v3', choices=['1v3', '2v2'])
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', default=None)
    ap.add_argument('--temperature', type=float, default=0.0,
                    help='>0 时对称套到候选+对手(@tT);=0 为 argmax 老路径')
    ap.add_argument('--opponents', default=None,
                    help='异质三对手 spec 逗号分隔(1v3),设置则覆盖 --opponent')
    ap.add_argument('--indep-opp', action='store_true',
                    help='三对手座位各自独立采样 rng(同质,共享 model);T>0 复式必需')
    ap.add_argument('--decision-salt', type=int, default=0,
                    help='改采样种子但不改墙,同墙换对手随机流复测取平均')
    args = ap.parse_args()

    t0 = time.time()
    records = []
    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.candidate, args.opponent, args.seed, args.mode,
                        args.temperature, args.opponents, args.indep_opp,
                        args.decision_salt)) as pool:
        for recs in pool.imap_unordered(_run_group, range(args.groups), chunksize=4):
            records.extend(recs)
    elapsed = time.time() - t0

    n = len(records)
    score_m, score_se = mean_se([r['score'] for r in records])
    hu = np.mean([r['hu'] for r in records])
    dianpao = np.mean([r['dianpao'] for r in records])
    zimo = np.mean([r['zimo'] for r in records])
    huang = np.mean([r['huang'] for r in records])
    fans = [r['fan'] for r in records if r['hu']]
    fan_m = float(np.mean(fans)) if fans else 0.0
    by_group = defaultdict(list)
    for r in records:
        by_group[r['group']].append(r['score'] - r['others_mean'])
    d_m, d_se = mean_se([np.mean(v) for v in by_group.values()])

    md = []
    md.append('# Arena: %s (x%d) vs %s — %s' % (args.candidate, 1 if args.mode == '1v3' else 2,
                                                args.opponent, args.mode))
    md.append('')
    md.append('- groups: %d, candidate games: %d, seed: %d, workers: %d, elapsed: %.1fs'
              % (args.groups, n, args.seed, args.workers, elapsed))
    if args.temperature or args.opponents or args.indep_opp or args.decision_salt:
        md.append('- temperature: %g, indep_opp: %s, opponents: %s, decision_salt: %d'
                  % (args.temperature, args.indep_opp, args.opponents, args.decision_salt))
    md.append('')
    md.append('| 指标 | 值 |')
    md.append('|---|---|')
    md.append('| 每局平均得分 | %.3f ± %.3f (SE) |' % (score_m, score_se))
    md.append('| 胡牌率 | %.2f%% |' % (hu * 100))
    md.append('| 自摸率 | %.2f%% |' % (zimo * 100))
    md.append('| 点炮率 | %.2f%% |' % (dianpao * 100))
    md.append('| 荒庄率 | %.2f%% |' % (huang * 100))
    md.append('| 胡牌平均番数 | %.2f |' % fan_m)
    md.append('| 同墙配对差值 | %.3f ± %.3f (SE, n=%d 组) |' % (d_m, d_se, len(by_group)))
    text = '\n'.join(md)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        with open(args.out.replace('.md', '.json'), 'w') as f:
            json.dump({'records': records, 'elapsed': elapsed}, f)


if __name__ == '__main__':
    main()
