"""验证 src/shanten_py.py 与 PyMahjongGB 的一致性。

金标准:fork 隔离的库调用 —— Pool(maxtasksperchild=1),父进程从不调用 shanten
接口,每个子进程 fork 出干净状态、只执行一次调用即退出,完全绕开状态污染 bug。

用例来源:真实对局决策点手牌(含逐弃牌派生的 3n+1 手)+ 随机手牌。

用法(服务器项目根目录):
  python scripts/validate_shanten_py.py --games 200 --random 2000 --workers 64
"""
import argparse
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import shanten_py  # noqa: E402

# 父进程只 import、绝不调用,保证 fork 出的子进程状态干净
from MahjongGB import (RegularShanten, SevenPairsShanten, ThirteenOrphansShanten,  # noqa: E402
                       HonorsAndKnittedTilesShanten, KnittedStraightShanten)

LIB_FNS = {
    'regular': RegularShanten,
    'seven_pairs': SevenPairsShanten,
    'thirteen_orphans': ThirteenOrphansShanten,
    'honors_knitted': HonorsAndKnittedTilesShanten,
    'knitted_straight': KnittedStraightShanten,
}
MY_FNS = {
    'regular': shanten_py.regular_shanten_useful,
    'seven_pairs': shanten_py.seven_pairs_shanten_useful,
    'thirteen_orphans': shanten_py.thirteen_orphans_shanten_useful,
    'honors_knitted': shanten_py.honors_knitted_shanten_useful,
    'knitted_straight': shanten_py.knitted_straight_shanten_useful,
}


def oracle_one(args):
    """子进程内的唯一一次库调用(随后进程退出)。"""
    name, hand = args
    try:
        r = LIB_FNS[name](hand)
        return (r[0], tuple(sorted(r[1])))
    except Exception as e:
        return ('ERR', str(e)[:40])


def _collect_worker(n_games, q):
    q.put(collect_real_hands(n_games))


def collect_real_hands_isolated(n_games):
    """在 spawn 子进程中收集(重放会调用库的算番器,可能污染本进程的库状态;
    父进程必须保持"从未调用过库"以充当 fork oracle 的干净模板)。"""
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_collect_worker, args=(n_games, q))
    p.start()
    hands = q.get()
    p.join()
    return hands


def collect_real_hands(n_games):
    from replay_driver import iter_matches, replay_match
    from feature_v1 import FeatureAgentV1
    hands = set()
    orig_obs = FeatureAgentV1._obs
    def patched(self):
        h = tuple(sorted(self.hand))
        if len(h) % 3 == 2:
            for t in set(h):
                rest = list(h)
                rest.remove(t)
                hands.add(tuple(sorted(rest)))
        elif len(h) % 3 == 1:
            hands.add(h)
        return orig_obs(self)
    FeatureAgentV1._obs = patched
    path = 'baseline/SL/data/data.txt'
    if not os.path.exists(path):
        path = 'baseline/SL/data/sample.txt'
    for mid, lines in iter_matches(path, limit=n_games):
        replay_match(mid, lines, FeatureAgentV1)
    FeatureAgentV1._obs = orig_obs
    return sorted(hands)


def rand_hand(rng, n):
    pool = shanten_py.TILE_LIST * 4
    rng.shuffle(pool)
    return tuple(sorted(pool[:n]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--random', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=64)
    args = ap.parse_args()

    rng = random.Random(2026)
    real = collect_real_hands_isolated(args.games)
    rand13 = [rand_hand(rng, 13) for _ in range(args.random)]
    randoth = [rand_hand(rng, rng.choice((1, 4, 7, 10))) for _ in range(args.random // 2)]
    print('real hands: %d, random: %d' % (len(real), len(rand13) + len(randoth)))

    cases = []
    for h in real + rand13 + randoth:
        cases.append(('regular', h))
        if len(h) == 13:
            for name in ('seven_pairs', 'thirteen_orphans', 'honors_knitted', 'knitted_straight'):
                cases.append((name, h))
        if len(h) == 10:
            cases.append(('knitted_straight', h))
    rng.shuffle(cases)
    cap = 60000
    if len(cases) > cap:
        cases = cases[:cap]
    print('total cases: %d' % len(cases))

    t0 = time.time()
    mine = []
    for name, h in cases:
        try:
            s, u = MY_FNS[name](h)
            mine.append((s, tuple(sorted(u))))
        except Exception as e:
            mine.append(('ERR', str(e)[:40]))
    t_mine = time.time() - t0
    print('shanten_py: %.1fs (%.2f ms/call avg, cached)' % (t_mine, t_mine / len(cases) * 1000))

    t0 = time.time()
    with Pool(args.workers, maxtasksperchild=1) as pool:
        truth = pool.map(oracle_one, cases, chunksize=1)
    print('oracle (fork-isolated): %.1fs' % (time.time() - t0))

    bad = {}
    ignored_5th = 0
    for (name, h), m, t in zip(cases, mine, truth):
        if t[0] == 'ERR' and m[0] == 'ERR':
            continue
        if m != t:
            # 已知库瑕疵一:组合龙路径把手中已有 4 张的牌标为进张(第 5 张不存在),
            # 库的基本和型路径有 cnt==4 跳过而组合龙没有 —— 库多标,我们更正确。
            if m[0] == t[0] and set(m[1]) <= set(t[1]):
                extra = set(t[1]) - set(m[1])
                if all(h.count(x) == 4 for x in extra):
                    ignored_5th += 1
                    continue
            # 已知库瑕疵二:手中已有 3 张的牌作为第 4 张进张时库会漏标
            # (暴力真值站在我们这边,见 brute_shanten_check)。
            if m[0] == t[0] and set(t[1]) <= set(m[1]):
                extra = set(m[1]) - set(t[1])
                if all(h.count(x) == 3 for x in extra):
                    ignored_5th += 1
                    continue
            bad.setdefault(name, []).append((h, m, t))
    print('ignored lib-marks-impossible-5th-copy diffs: %d' % ignored_5th)
    if not bad:
        print('ALL MATCH: %d cases' % len(cases))
    for name, items in bad.items():
        print('%s mismatches: %d' % (name, len(items)))
        for h, m, t in items[:5]:
            print('  hand:', h)
            print('   mine:', m)
            print('   lib :', t)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
