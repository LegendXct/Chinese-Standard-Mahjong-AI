"""诊断/验证 PyMahjongGB shanten 接口的调用序列污染 bug 及"重复调用至稳定"规避方案。

已知现象:RegularShanten(X) 的结果可能受上一次调用污染(错误),
对同一 X 立即重复调用会回到正确值。本脚本:
  1) 随机调用序列中统计 第1次结果 != 第2次结果 的频率(污染发生率);
  2) 验证稳定值(连续两次相同)== 金标准(全新进程中的首次调用);
  3) 同时覆盖 RegularShanten 与特殊牌型接口。

用法:python scripts/diagnose_shanten.py --seqs 200 --calls 40 --oracle-sample 300
"""
import argparse
import json
import random
import subprocess
import sys

from MahjongGB import (RegularShanten, SevenPairsShanten, ThirteenOrphansShanten,
                       HonorsAndKnittedTilesShanten, KnittedStraightShanten)

TILES = (['W%d' % i for i in range(1, 10)] + ['T%d' % i for i in range(1, 10)] +
         ['B%d' % i for i in range(1, 10)] + ['F%d' % i for i in range(1, 5)] +
         ['J%d' % i for i in range(1, 4)])

FNS = {
    'RegularShanten': (RegularShanten, (1, 4, 7, 10, 13)),
    'SevenPairsShanten': (SevenPairsShanten, (13,)),
    'ThirteenOrphansShanten': (ThirteenOrphansShanten, (13,)),
    'HonorsAndKnittedTilesShanten': (HonorsAndKnittedTilesShanten, (13,)),
    'KnittedStraightShanten': (KnittedStraightShanten, (13,)),
}


def rand_hand(rng, n):
    pool = TILES * 4
    rng.shuffle(pool)
    return tuple(sorted(pool[:n]))


def norm(r):
    return (r[0], tuple(sorted(r[1])))


def call(fn, hand):
    try:
        return norm(fn(hand))
    except Exception as e:
        return ('ERR', str(e))


def oracle(fn_name, hand):
    """全新进程中的首次调用 = 金标准。"""
    code = ('from MahjongGB import %s as F\n'
            'r = F(%r)\n'
            'print(repr((r[0], tuple(sorted(r[1])))))' % (fn_name, hand))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    if out.returncode != 0:
        return ('ERR', out.stderr.strip()[-80:])
    return eval(out.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', type=int, default=200)
    ap.add_argument('--calls', type=int, default=40)
    ap.add_argument('--oracle-sample', type=int, default=300)
    args = ap.parse_args()

    n_calls = 0
    unstable = []          # (fn_name, hand, r1, r_stable)
    stable_pool = []       # (fn_name, hand, r_stable)
    not_converged = 0

    def probe(fn_name, fn, hand):
        nonlocal n_calls, not_converged
        r1 = call(fn, hand)
        r2 = call(fn, hand)
        n_calls += 1
        if r1 != r2:
            r3 = call(fn, hand)
            if r2 != r3:
                not_converged += 1
                unstable.append((fn_name, hand, r1, r3))
            else:
                unstable.append((fn_name, hand, r1, r2))
        else:
            stable_pool.append((fn_name, hand, r1))

    for seq in range(args.seqs):
        rng = random.Random(1000 + seq)
        # 模式 A:独立随机手牌
        for _ in range(args.calls // 2):
            fn_name = rng.choice(list(FNS))
            fn, sizes = FNS[fn_name]
            probe(fn_name, fn, rand_hand(rng, rng.choice(sizes)))
        # 模式 B(真实使用模式):同一手牌的逐弃牌枚举 —— 相邻调用只差一张牌
        full = rand_hand(rng, rng.choice((14, 11, 8)))
        for t in sorted(set(full)):
            rest = list(full)
            rest.remove(t)
            probe('RegularShanten', RegularShanten, tuple(sorted(rest)))
            if len(rest) == 13:
                for fn_name in ('SevenPairsShanten', 'ThirteenOrphansShanten',
                                'HonorsAndKnittedTilesShanten', 'KnittedStraightShanten'):
                    probe(fn_name, FNS[fn_name][0], tuple(sorted(rest)))

    print('total calls: %d, polluted(1st!=2nd): %d (%.3f%%), not converged by 3rd: %d'
          % (n_calls, len(unstable), 100.0 * len(unstable) / n_calls, not_converged))
    by_fn = {}
    for fn_name, _, _, _ in unstable:
        by_fn[fn_name] = by_fn.get(fn_name, 0) + 1
    print('polluted by fn:', json.dumps(by_fn))

    # 金标准验证:所有污染样本的稳定值 + 随机抽样的"看似稳定"样本
    rng = random.Random(7)
    checks = [(f, h, s) for f, h, _, s in unstable]
    checks += rng.sample(stable_pool, min(args.oracle_sample, len(stable_pool)))
    bad_stable = 0
    for fn_name, hand, stable in checks:
        truth = oracle(fn_name, hand)
        if truth != stable:
            bad_stable += 1
            print('STABLE != ORACLE:', fn_name, hand, 'stable=', stable, 'oracle=', truth)
    print('oracle checks: %d, stable-value mismatches: %d' % (len(checks), bad_stable))
    if unstable[:5]:
        print('examples of pollution:')
        for fn_name, hand, r1, rs in unstable[:5]:
            print(' ', fn_name, hand, '1st=', r1, 'stable=', rs)
    sys.exit(1 if bad_stable else 0)


if __name__ == '__main__':
    main()
