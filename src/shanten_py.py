"""纯 Python 向听数/有效进张引擎(替代 PyMahjongGB 的 shanten 接口)。

背景:PyMahjongGB 1.3.0 wheel 的 shanten 系列接口存在跨调用状态污染
(疑似静态工作缓冲区未复位),返回值依赖调用历史且无法在 Python 层可靠规避
(见 scripts/diagnose_shanten.py)。算番器 MahjongFanCalculator 不受影响
(98209 局 / 1847 万决策点重放 0 违规),继续使用库;向听走本模块。

接口约定与库一致:
  regular_shanten_useful(hand)            基本和型
  seven_pairs_shanten_useful(hand)        七对(13 张)
  thirteen_orphans_shanten_useful(hand)   十三幺(13 张)
  honors_knitted_shanten_useful(hand)     全不靠(13 张)
  knitted_straight_shanten_useful(hand)   组合龙(13/10 张)
hand 为牌名字符串 tuple(任意顺序),返回 (shanten, frozenset(useful))。
正确性由 scripts/validate_shanten_py.py 以"fork 隔离的库首次调用"为金标准验证。
"""
import os
from functools import lru_cache

# 缓存条目数(内存敏感:容器 cgroup 限 90GB,多进程下每 worker 独立持有缓存)
_CACHE_L1 = int(os.environ.get('SHANTEN_CACHE_L1', 131072))   # _suit_options
_CACHE_L2 = int(os.environ.get('SHANTEN_CACHE_L2', 262144))   # _regular_shanten_counts

TILE_LIST = [
    *('W%d' % (i + 1) for i in range(9)),
    *('T%d' % (i + 1) for i in range(9)),
    *('B%d' % (i + 1) for i in range(9)),
    *('F%d' % (i + 1) for i in range(4)),
    *('J%d' % (i + 1) for i in range(3))
]
OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}
_MAJORS = [OFFSET_TILE[t] for t in
           ('W1', 'W9', 'T1', 'T9', 'B1', 'B9', 'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3')]
_HONORS = list(range(27, 34))
# 组合龙的 6 种花色指派:({1,4,7}, {2,5,8}, {3,6,9}) 的全排列落到 (W,T,B)
_KNIT_RESIDUES = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]
_KNIT_PATTERNS = []
for a in range(3):
    for b in range(3):
        if b == a:
            continue
        c = 3 - a - b
        pat = []
        for suit, res in ((0, a), (1, b), (2, c)):
            pat.extend(suit * 9 + r for r in _KNIT_RESIDUES[res])
        _KNIT_PATTERNS.append(tuple(sorted(pat)))


def counts_from_hand(hand):
    c = [0] * 34
    for t in hand:
        c[OFFSET_TILE[t]] += 1
    return c


def _tiles_from(idx_set):
    return frozenset(TILE_LIST[i] for i in idx_set)


# ---------------- 基本和型 ----------------
def _pareto(entries):
    """保留 (sets, partials, pairs) 的非被支配项。"""
    out = []
    for e in entries:
        if any(o != e and o[0] >= e[0] and o[1] >= e[1] and o[2] >= e[2] for o in entries):
            continue
        out.append(e)
    return frozenset(out)


@lru_cache(maxsize=_CACHE_L1)
def _suit_options(tiles, allow_runs):
    """tiles: 长度 9(花色)或 7(字牌)的计数 tuple。
    返回 Pareto 集 {(sets, partials, pairs)},partials 为两张搭子(不含对子)。"""
    n = len(tiles)
    i = -1
    for k in range(n):
        if tiles[k]:
            i = k
            break
    if i < 0:
        return frozenset({(0, 0, 0)})
    res = set()
    c = list(tiles)
    # 该张作孤张
    c[i] -= 1
    res |= _suit_options(tuple(c), allow_runs)
    c[i] += 1
    # 刻子
    if c[i] >= 3:
        c[i] -= 3
        res |= {(s + 1, p, q) for s, p, q in _suit_options(tuple(c), allow_runs)}
        c[i] += 3
    # 对子
    if c[i] >= 2:
        c[i] -= 2
        res |= {(s, p, q + 1) for s, p, q in _suit_options(tuple(c), allow_runs)}
        c[i] += 2
    if allow_runs:
        # 顺子
        if i + 2 < n and c[i + 1] and c[i + 2]:
            c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
            res |= {(s + 1, p, q) for s, p, q in _suit_options(tuple(c), allow_runs)}
            c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
        # 两面/边张搭子
        if i + 1 < n and c[i + 1]:
            c[i] -= 1; c[i + 1] -= 1
            res |= {(s, p + 1, q) for s, p, q in _suit_options(tuple(c), allow_runs)}
            c[i] += 1; c[i + 1] += 1
        # 嵌张搭子
        if i + 2 < n and c[i + 2]:
            c[i] -= 1; c[i + 2] -= 1
            res |= {(s, p + 1, q) for s, p, q in _suit_options(tuple(c), allow_runs)}
            c[i] += 1; c[i + 2] += 1
    return _pareto(res)


@lru_cache(maxsize=_CACHE_L2)
def _regular_shanten_counts(counts, need):
    """counts: 34 维计数 tuple;need: 还需要的面子数(4 - 副露数)。
    手牌张数应为 3*need+1 或 3*need+2。返回向听数(和牌张为 -1)。
    标量查询走递归+缓存(带缓存的递归本身就是查表,单次更快);
    批量探测(useful 计算)走 shanten_table 的向量化路径,见 regular_shanten_useful。"""
    return _regular_shanten_counts_recursive(counts, need)


def _regular_shanten_counts_recursive(counts, need):
    folded = {(0, 0, 0)}
    for lo, hi, runs in ((0, 9, True), (9, 18, True), (18, 27, True), (27, 34, False)):
        opts = _suit_options(tuple(counts[lo:hi]), runs)
        folded = {(s1 + s2, p1 + p2, q1 + q2)
                  for s1, p1, q1 in folded for s2, p2, q2 in opts}
        # 折叠途中粗剪枝(保守上限,避免组合爆炸)
        folded = {(min(s, need), min(p, need + 1), min(q, need + 1)) for s, p, q in folded}
    best = 99
    for s, p, q in folded:
        m = min(s, need)
        for use_pair in ((0, 1) if q >= 1 else (0,)):
            d = min(p + (q - use_pair), need - m)
            sh = 2 * need - 2 * m - d - use_pair
            if sh < best:
                best = sh
    return best


def _batch_regular_table(tab, counts_mat, need):
    """向量化批量向听:counts_mat (N,34) -> (N,) 向听数。与递归严格等价。"""
    import numpy as _np
    import shanten_table as _st
    P9 = _np.array(_st.POW5_9, _np.int64)
    P7 = _np.array(_st.POW5_7, _np.int64)
    idx = [counts_mat[:, 0:9] @ P9, counts_mat[:, 9:18] @ P9,
           counts_mat[:, 18:27] @ P9, counts_mat[:, 27:34] @ P7]
    N = len(counts_mat)
    NEGV = -100
    dp = _np.full((N, need + 1, 2), NEGV, _np.int16)
    dp[:, 0, 0] = 0
    tables = [(tab.sA, tab.sB), (tab.sA, tab.sB), (tab.sA, tab.sB), (tab.hA, tab.hB)]
    for g, (TA, TB) in enumerate(tables):
        A = _np.asarray(TA[idx[g]], _np.int16)   # (N,5)
        B = _np.asarray(TB[idx[g]], _np.int16)
        ndp = _np.full_like(dp, NEGV)
        for u in range(need + 1):
            kmax = min(4, need - u)
            for k in range(kmax + 1):
                cand = dp[:, u, :] + A[:, k:k + 1]
                _np.maximum(ndp[:, u + k, :], cand, out=ndp[:, u + k, :])
                bk = B[:, k]
                cand_b = _np.where(bk >= 0, dp[:, u, 0] + bk, NEGV)
                _np.maximum(ndp[:, u + k, 1], cand_b, out=ndp[:, u + k, 1])
        dp = ndp
    best = _np.maximum(dp[:, :, 0].max(axis=1), dp[:, :, 1].max(axis=1) + 1)
    return 2 * need - best


def regular_shanten_useful(hand):
    counts = counts_from_hand(hand)
    n = len(hand)
    if n not in (1, 4, 7, 10, 13):
        raise ValueError('regular_shanten_useful expects 3n+1 hand, got %d' % n)
    need = 4 - (13 - n) // 3
    import shanten_table
    tab = shanten_table.get_table()
    if tab is not None:
        import numpy as _np
        base_row = _np.array(counts, _np.int64)
        probes = [t for t in range(34) if counts[t] < 4]
        mat = _np.tile(base_row, (1 + len(probes), 1))
        for i, t in enumerate(probes):
            mat[1 + i, t] += 1
        sh = _batch_regular_table(tab, mat, need)
        base = int(sh[0])
        useful = {t for i, t in enumerate(probes) if sh[1 + i] < base}
        return base, _tiles_from(useful)
    base = _regular_shanten_counts(tuple(counts), need)
    useful = set()
    for t in range(34):
        if counts[t] >= 4:
            continue
        counts[t] += 1
        if _regular_shanten_counts(tuple(counts), need) < base:
            useful.add(t)
        counts[t] -= 1
    return base, _tiles_from(useful)


# ---------------- 七对(13 张) ----------------
def seven_pairs_shanten_useful(hand):
    if len(hand) != 13:
        raise ValueError('seven pairs requires 13 tiles')
    counts = counts_from_hand(hand)
    pairs = sum(c // 2 for c in counts)
    shanten = 6 - pairs
    useful = {t for t in range(34) if counts[t] % 2 == 1}
    return shanten, _tiles_from(useful)


# ---------------- 十三幺(13 张) ----------------
def thirteen_orphans_shanten_useful(hand):
    if len(hand) != 13:
        raise ValueError('thirteen orphans requires 13 tiles')
    counts = counts_from_hand(hand)
    kinds = sum(1 for t in _MAJORS if counts[t])
    has_pair = any(counts[t] >= 2 for t in _MAJORS)
    shanten = 13 - kinds - (1 if has_pair else 0)
    missing = {t for t in _MAJORS if counts[t] == 0}
    if has_pair:
        useful = missing
    else:
        useful = missing | {t for t in _MAJORS if counts[t]}  # 已有幺九成对也减向听
    return shanten, _tiles_from(useful)


# ---------------- 全不靠(13 张) ----------------
def honors_knitted_shanten_useful(hand):
    if len(hand) != 13:
        raise ValueError('honors and knitted requires 13 tiles')
    counts = counts_from_hand(hand)
    best = 99
    useful = set()
    for pat in _KNIT_PATTERNS:
        usable = list(pat) + _HONORS
        kinds = sum(1 for t in usable if counts[t])
        sh = 13 - kinds
        miss = {t for t in usable if counts[t] == 0}
        if sh < best:
            best, useful = sh, set(miss)
        elif sh == best:
            useful |= miss
    return best, _tiles_from(useful)


# ---------------- 组合龙(13/10 张) ----------------
def knitted_straight_shanten_useful(hand):
    n = len(hand)
    if n not in (13, 10):
        raise ValueError('knitted straight requires 13 or 10 tiles')
    counts = counts_from_hand(hand)
    need_rest = 1 if n == 13 else 0  # 组合龙占 3 个面子位,其余部分还需的面子数
    import shanten_table
    tab = shanten_table.get_table()
    best = 99
    useful = set()
    for pat in _KNIT_PATTERNS:
        rest = list(counts)
        k = 0
        miss_knit = set()
        for t in pat:
            if rest[t]:
                rest[t] -= 1
                k += 1
            else:
                miss_knit.add(t)
        # useful = 缺的组合龙张 + 使剩余部分降向听的张
        # (模式内的牌也可能服务于剩余部分,如给对子凑张,不可跳过)
        if tab is not None:
            import numpy as _np
            probes = [t for t in range(34) if rest[t] < 4]
            mat = _np.tile(_np.array(rest, _np.int64), (1 + len(probes), 1))
            for i, t in enumerate(probes):
                mat[1 + i, t] += 1
            shv = _batch_regular_table(tab, mat, need_rest)
            sub = int(shv[0])
            sh = (9 - k) + sub
            if sh > best:
                continue
            sub_useful = {t for i, t in enumerate(probes) if shv[1 + i] < sub}
        else:
            sub = _regular_shanten_counts(tuple(rest), need_rest)
            sh = (9 - k) + sub
            if sh > best:
                continue
            sub_useful = set()
            for t in range(34):
                if rest[t] >= 4:
                    continue
                rest[t] += 1
                if _regular_shanten_counts(tuple(rest), need_rest) < sub:
                    sub_useful.add(t)
                rest[t] -= 1
        if sh < best:
            best, useful = sh, miss_knit | sub_useful
        else:
            useful |= miss_knit | sub_useful
    return best, _tiles_from(useful)
