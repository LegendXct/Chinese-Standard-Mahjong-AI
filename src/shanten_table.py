"""向听查表化:离线预分解 numpy 表 + 快速查询(预案,M3 触发)。

表结构(int8,np.save 落盘,运行时 np.load(mmap_mode='r') 只读共享页缓存):
  suit_A[idx, k]  花色向量 idx(9 格计数 base-5 编码)在块预算 k(0..4)下,
                  不保留对子时 max(2*面子 + 搭子) ;搭子池含"多余对子当搭子"
  suit_B[idx, k]  保留一个对子作将(要求至少有一对)后的同值;无对子处为 -1
  honor_A/B       字牌 7 格向量(5^7),无顺子/搭子(只刻子/对子)
查询:regular_shanten_counts(counts34, need) =
  2*need - max over 4 组块分配与至多一组留将 of Σ值 (+1 若留将)
与 shanten_py 的 Pareto 递归严格等价(见 --verify)。

用法:
  python src/shanten_table.py --build --out /root/autodl-tmp/mahjong/shanten_tables
  python src/shanten_table.py --verify --out ... --cases 20000
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

POW5_9 = [5 ** i for i in range(9)]
POW5_7 = [5 ** i for i in range(7)]


def _entries_to_AB(entries, n_blocks=5):
    """Pareto 集 {(s,p,q)} -> (A[5], B[5])。"""
    A = np.full(n_blocks, -1, np.int8)
    B = np.full(n_blocks, -1, np.int8)
    for k in range(n_blocks):
        best_a = best_b = -1
        for s, p, q in entries:
            m = min(s, k)
            d = min(p + q, k - m)
            best_a = max(best_a, 2 * m + d)
            if q >= 1:
                d = min(p + q - 1, k - m)
                best_b = max(best_b, 2 * m + d)
        A[k] = best_a
        B[k] = best_b
    return A, B


def build(out_dir):
    from shanten_py import _suit_options
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for name, n_cells, allow_runs in (('suit', 9, True), ('honor', 7, False)):
        n = 5 ** n_cells
        A = np.full((n, 5), -1, np.int8)
        B = np.full((n, 5), -1, np.int8)
        cnt = 0
        for vec in itertools.product(range(5), repeat=n_cells):
            if sum(vec) > 14:
                continue
            idx = sum(v * 5 ** i for i, v in enumerate(vec))
            entries = _suit_options(tuple(vec), allow_runs)
            A[idx], B[idx] = _entries_to_AB(entries)
            cnt += 1
            if cnt % 200000 == 0:
                print('%s %d done %.1fs' % (name, cnt, time.time() - t0), flush=True)
        np.save(os.path.join(out_dir, '%s_A.npy' % name), A)
        np.save(os.path.join(out_dir, '%s_B.npy' % name), B)
        print('%s: %d valid vectors, %.1fs' % (name, cnt, time.time() - t0), flush=True)


class TableShanten:

    def __init__(self, out_dir):
        self.sA = np.load(os.path.join(out_dir, 'suit_A.npy'), mmap_mode='r')
        self.sB = np.load(os.path.join(out_dir, 'suit_B.npy'), mmap_mode='r')
        self.hA = np.load(os.path.join(out_dir, 'honor_A.npy'), mmap_mode='r')
        self.hB = np.load(os.path.join(out_dir, 'honor_B.npy'), mmap_mode='r')

    @staticmethod
    def _indices(counts):
        i0 = sum(counts[j] * POW5_9[j] for j in range(9))
        i1 = sum(counts[9 + j] * POW5_9[j] for j in range(9))
        i2 = sum(counts[18 + j] * POW5_9[j] for j in range(9))
        i3 = sum(counts[27 + j] * POW5_7[j] for j in range(7))
        return i0, i1, i2, i3

    def shanten(self, counts, need):
        i0, i1, i2, i3 = self._indices(counts)
        rows = ((self.sA[i0], self.sB[i0]), (self.sA[i1], self.sB[i1]),
                (self.sA[i2], self.sB[i2]), (self.hA[i3], self.hB[i3]))
        # DP over (已用块数, 是否已留将) -> max 值
        NEG = -100
        dp = [[NEG] * 2 for _ in range(need + 1)]
        dp[0][0] = 0
        for Arow, Brow in rows:
            ndp = [[NEG] * 2 for _ in range(need + 1)]
            for used in range(need + 1):
                for pu in range(2):
                    base = dp[used][pu]
                    if base <= NEG:
                        continue
                    kmax = min(4, need - used)
                    for k in range(kmax + 1):
                        a = Arow[k]
                        if base + a > ndp[used + k][pu]:
                            ndp[used + k][pu] = base + a
                        if pu == 0:
                            b = Brow[k]
                            if b >= 0 and base + b > ndp[used + k][1]:
                                ndp[used + k][1] = base + b
            dp = ndp
        best = -1
        for used in range(need + 1):
            if dp[used][0] > best:
                best = dp[used][0]
            if dp[used][1] + 1 > best:
                best = dp[used][1] + 1
        return 2 * need - best


_TABLE = None


def get_table():
    """显式 opt-in:仅当设置 SHANTEN_TABLE_DIR 时启用批量表路径。
    实测(M3.0 复测)记忆化递归因跨查询缓存复用反而更快,故默认关闭,
    表实现与等价验证保留备用。"""
    global _TABLE
    if _TABLE is None:
        d = os.environ.get('SHANTEN_TABLE_DIR')
        if d and os.path.exists(os.path.join(d, 'suit_A.npy')):
            _TABLE = TableShanten(d)
        else:
            _TABLE = False
    return _TABLE or None


def verify(out_dir, n_cases, seed=11):
    import random
    from shanten_py import _regular_shanten_counts_recursive as _rec
    tab = TableShanten(out_dir)
    rng = random.Random(seed)
    t_tab = t_rec = 0.0
    for c in range(n_cases):
        n = rng.choice((1, 4, 7, 10, 13, 2, 5, 8, 11, 14))
        pool = list(range(34)) * 4
        rng.shuffle(pool)
        counts = [0] * 34
        for t in pool[:n]:
            counts[t] += 1
        need = 4 - (13 - n) // 3 if n % 3 == 1 else 4 - (14 - n) // 3
        t0 = time.perf_counter()
        s_tab = tab.shanten(counts, need)
        t_tab += time.perf_counter() - t0
        t0 = time.perf_counter()
        s_rec = _rec(tuple(counts), need)
        t_rec += time.perf_counter() - t0
        assert s_tab == s_rec, 'mismatch counts=%s need=%d tab=%d rec=%d' % (
            counts, need, s_tab, s_rec)
    print('VERIFY PASS: %d cases, table %.1f us/call vs recursion %.1f us/call' % (
        n_cases, t_tab / n_cases * 1e6, t_rec / n_cases * 1e6))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--out', default='/root/autodl-tmp/mahjong/shanten_tables')
    ap.add_argument('--cases', type=int, default=20000)
    args = ap.parse_args()
    if args.build:
        build(args.out)
    if args.verify:
        verify(args.out, args.cases)
