"""3d 暴力真值抽检:用与向听引擎完全独立的方法验证 shanten_py 的向听数。

方法:以 MahjongFanCalculator 作和牌结构判定器(已被 1847 万决策点重放认证,
且无已知状态污染),做限深换牌搜索:
  向听 0(听牌) <=> 存在摸牌使手牌成和
  向听 1        <=> 非听牌,且存在一次"弃一摸一"后听牌
  向听 2        <=> 非听牌、一次换牌不可达听牌,且存在两次换牌路径(用
                    shanten_py 的进张做引导寻路,找到即构成存在性证明)
三方比对:shanten_py / fork 隔离的库干净首调(MahjongShanten)/ 暴力真值。

用法:python src/brute_shanten_check.py --games 60 --sample 200 --workers 32
"""
import argparse
import os
import random
import subprocess
import sys
import time
from functools import lru_cache
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shanten_py  # noqa: E402
from feature_v1 import FeatureAgentV1, shanten_useful_wait  # noqa: E402
from replay_driver import iter_matches, replay_match  # noqa: E402
from MahjongGB import MahjongFanCalculator  # noqa: E402

TILES = shanten_py.TILE_LIST


def _is_win(packs, hand, winTile):
    try:
        MahjongFanCalculator(pack=packs, hand=tuple(hand), winTile=winTile,
                             flowerCount=0, isSelfDrawn=True, is4thTile=False,
                             isAboutKong=False, isWallLast=False, seatWind=0,
                             prevalentWind=0, verbose=True)
        return True
    except Exception:
        return False


def _tenpai(packs, hand_sorted):
    """听牌判定 + 听的牌集合。memoize 见 _tenpai_cached。"""
    waits = []
    for t in TILES:
        if hand_sorted.count(t) >= 4:
            continue
        if _is_win(packs, hand_sorted, t):
            waits.append(t)
    return waits


def brute_check_one(args):
    """返回 (claimed, brute_ok, lib_clean, hand, packs)"""
    packs, hand = args
    hand = tuple(sorted(hand))
    claimed, useful = shanten_useful_wait(hand)

    @lru_cache(maxsize=None)
    def tenpai(h):
        return tuple(_tenpai(packs, h))

    def swaps(h):
        for d in sorted(set(h)):
            rest = list(h)
            rest.remove(d)
            for a in TILES:
                if rest.count(a) >= 4 or a == d:
                    continue
                yield tuple(sorted(rest + [a]))

    if claimed == 0:
        ok = bool(tenpai(hand))
    elif claimed == 1:
        ok = (not tenpai(hand)) and any(tenpai(h) for h in swaps(hand))
    elif claimed == 2:
        not_0 = not tenpai(hand)
        not_1 = not any(tenpai(h) for h in swaps(hand))
        # 引导寻路证明可达性:换一张后 shanten_py 称 1,再暴力验证该 1
        path2 = False
        for h1 in swaps(hand):
            if shanten_useful_wait(h1)[0] == 1 and not tenpai(h1) \
                    and any(tenpai(h2) for h2 in swaps(h1)):
                path2 = True
                break
        ok = not_0 and not_1 and path2
    else:
        return None
    return claimed, ok, hand, packs


def lib_clean_shanten(packs, hand):
    """fork/spawn 隔离的库干净首调(MahjongShanten 综合向听)。"""
    code = ('from MahjongGB import MahjongShanten\n'
            'print(MahjongShanten(pack=%r, hand=%r))' % (tuple(packs), tuple(hand)))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def collect(n_games, sample_n, seed):
    """收集 claimed 向听 <=2 的 (packs, hand) 决策点样本。"""
    pool = []
    orig = FeatureAgentV1._obs
    def patched(self):
        if len(self.hand) % 3 == 1:
            pool.append((tuple(self.packs[0]), tuple(sorted(self.hand))))
        return None
    FeatureAgentV1._obs = patched
    path = 'baseline/SL/data/data.txt'
    if not os.path.exists(path):
        path = 'baseline/SL/data/sample.txt'
    for mid, lines in iter_matches(path, limit=n_games):
        replay_match(mid, lines, FeatureAgentV1)
    FeatureAgentV1._obs = orig
    rng = random.Random(seed)
    rng.shuffle(pool)
    out = []
    seen = set()
    for packs, hand in pool:
        if (packs, hand) in seen:
            continue
        seen.add((packs, hand))
        # 他家暗杠 CONCEALED 不会出现在自家 packs;防御性过滤
        if any(t == 'CONCEALED' for _, t, _ in packs):
            continue
        if shanten_useful_wait(hand)[0] <= 2:
            out.append((packs, hand))
        if len(out) >= sample_n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=60)
    ap.add_argument('--sample', type=int, default=200)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    t0 = time.time()
    cases = collect(args.games, args.sample, args.seed)
    print('cases: %d (claimed shanten<=2), collect %.1fs' % (len(cases), time.time() - t0))

    t1 = time.time()
    with Pool(args.workers) as pool:
        brute = pool.map(brute_check_one, cases)
    print('brute: %.1fs' % (time.time() - t1))

    t2 = time.time()
    lib = [lib_clean_shanten(p, h) for p, h in cases]
    print('lib clean first-call: %.1fs' % (time.time() - t2))

    n_brute_ok = sum(1 for b in brute if b and b[1])
    n_lib_ok = sum(1 for (p, h), s in zip(cases, lib)
                   if s == shanten_useful_wait(tuple(sorted(h)))[0])
    print('shanten_py vs 暴力真值: %d/%d 一致' % (n_brute_ok, len(cases)))
    print('shanten_py vs 库干净首调: %d/%d 一致' % (n_lib_ok, len(cases)))
    bad = [b for b in brute if b and not b[1]]
    for b in bad[:5]:
        print('  BRUTE-MISMATCH claimed=%d hand=%s packs=%s' % (b[0], b[2], b[3]))
    for (p, h), s in list(zip(cases, lib))[:200]:
        c = shanten_useful_wait(tuple(sorted(h)))[0]
        if s is not None and s != c:
            print('  LIB-MISMATCH lib=%d py=%d hand=%s packs=%s' % (s, c, h, p))
    sys.exit(1 if (n_brute_ok < len(cases) or n_lib_ok < len(cases)) else 0)


if __name__ == '__main__':
    main()
