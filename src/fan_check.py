"""3a 番数全量对拍:重建每局和牌时刻的状态,按 feature 侧旗标语义重算总番,
与牌谱 Fan <F> 行逐局比对。官方裁判的实际输出(Fan 行)是金标准。

不一致样本自动归因:对 is4thTile(和绝张)/isAboutKong(杠上开花·抢杠)/
isWallLast(妙手回春·海底捞月)逐个翻转重算,看哪个翻转能复现日志番数。

用法(服务器项目根目录):
  python src/fan_check.py --data baseline/SL/data/data.txt --workers 96
  python src/fan_check.py --data baseline/SL/data/sample.txt --workers 4
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_v1 import FeatureAgentV1, judge_fan_total  # noqa: E402
from preprocess_v1 import index_match_offsets, _iter_matches_in_slice  # noqa: E402
from MahjongGB import MahjongFanCalculator  # noqa: E402


def _calc_fan(packs, hand, winTile, isSelfDrawn, is4th, isAboutKong, isWallLast,
              seatWind, prevalentWind):
    try:
        fans = MahjongFanCalculator(
            pack=tuple(packs), hand=tuple(hand), winTile=winTile, flowerCount=0,
            isSelfDrawn=isSelfDrawn, is4thTile=is4th, isAboutKong=isAboutKong,
            isWallLast=isWallLast, seatWind=seatWind, prevalentWind=prevalentWind,
            verbose=True)
        return judge_fan_total(fans, winTile, hand), [(f, c, n) for f, c, n, _ in fans]
    except Exception as e:
        return None, str(e)[:60]


def check_match(match_id, lines):
    """重放单局;若以 Hu 结束返回比对结果 dict,否则 None。

    驱动事件与 replay_driver 相同,但 patch 掉 _obs 渲染(无需特征,省向听计算)。
    """
    agents = None
    curTile = None
    prev_event = None     # ('draw', p) / ('play', p) / ('bugang', p)
    win = None
    logged_fan = None

    for line in lines:
        t = line.split()
        if not t:
            continue
        if t[0] == 'Wind':
            agents = [FeatureAgentV1(i) for i in range(4)]
            for a in agents:
                a.request2obs(line)
        elif t[0] == 'Player':
            p = int(t[1])
            ev = t[2]
            if ev == 'Deal':
                agents[p].request2obs(' '.join(t[2:]))
            elif ev == 'Draw':
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(' '.join(t[2:4]))
                    else:
                        agents[i].request2obs(' '.join(t[:3]))
                prev_event = ('draw', p)
            elif ev == 'Play':
                for i in range(4):
                    agents[i].request2obs(' '.join(t[:4]))
                curTile = t[3]
                prev_event = ('play', p)
            elif ev == 'Chi':
                for i in range(4):
                    agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
                prev_event = ('chi', p)
            elif ev == 'Peng':
                for i in range(4):
                    agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
                prev_event = ('peng', p)
            elif ev == 'Gang':
                for i in range(4):
                    agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
                prev_event = ('gang', p)
            elif ev == 'AnGang':
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                    else:
                        agents[i].request2obs('Player %d AnGang' % p)
                prev_event = ('angang', p)
            elif ev == 'BuGang':
                for i in range(4):
                    agents[i].request2obs('Player %d BuGang %s' % (p, t[3]))
                curTile = t[3]
                prev_event = ('bugang', p)
            elif ev == 'Hu':
                win = (p, t[3], prev_event)
        elif t[0] == 'Fan':
            logged_fan = int(t[1])
        # Score/Huang 行无需处理

    if win is None:
        return None
    p, winTile, prev = win
    w = agents[p]
    if prev[0] == 'draw' and prev[1] == p:
        mode = 'zimo'
        isSelfDrawn = True
        hand = list(w.hand)
        hand.remove(winTile)
        isAboutKong = w._aboutKongRender
    elif prev[0] == 'bugang':
        mode = 'rob'
        isSelfDrawn = False
        hand = list(w.hand)
        isAboutKong = True
    else:
        mode = 'discard'
        isSelfDrawn = False
        hand = list(w.hand)
        isAboutKong = False
    # 抢杠和:官方裁判从不计和绝张(BUGANG 时 shownTile 已置 4,judge ==3 永假)
    is4th = False if mode == 'rob' else (w.shownTiles[winTile] + isSelfDrawn) == 4
    base_args = dict(packs=w.packs[0], hand=hand, winTile=winTile,
                     isSelfDrawn=isSelfDrawn, is4th=is4th, isAboutKong=isAboutKong,
                     isWallLast=w.wallLast, seatWind=p, prevalentWind=w.prevalentWind)
    fan, detail = _calc_fan(**base_args)
    out = {'match_id': match_id, 'mode': mode, 'logged': logged_fan, 'calc': fan,
           'flags': {'is4th': is4th, 'isAboutKong': isAboutKong, 'isWallLast': w.wallLast}}
    if fan == logged_fan:
        out['ok'] = True
        return out
    # 归因:逐个翻转旗标重算
    out['ok'] = False
    out['detail'] = detail
    out['toggles'] = {}
    for key in ('is4th', 'isAboutKong', 'isWallLast'):
        args2 = dict(base_args)
        args2[key] = not args2[key]
        f2, _ = _calc_fan(**args2)
        out['toggles'][key] = f2
        if f2 == logged_fan:
            out['explained_by'] = key
    return out


def check_chunk(task):
    chunk_id, path, start, end = task
    results = []
    no_op = lambda self: None
    orig = FeatureAgentV1._obs
    FeatureAgentV1._obs = no_op  # 跳过特征渲染(省向听计算,状态机不受影响)
    try:
        for match_id, lines in _iter_matches_in_slice(path, start, end):
            r = check_match(match_id, lines)
            if r is not None:
                results.append(r)
    finally:
        FeatureAgentV1._obs = orig
    return chunk_id, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--out', default='results/fan_check.json')
    args = ap.parse_args()

    t0 = time.time()
    offsets, fsize = index_match_offsets(args.data)
    n_games = len(offsets)
    n_chunks = min(args.workers * 4, n_games)
    bnd = np.linspace(0, n_games, n_chunks + 1).astype(int)
    tasks = []
    for c in range(n_chunks):
        lo, hi = bnd[c], bnd[c + 1]
        if lo == hi:
            continue
        tasks.append((c, args.data, offsets[lo], offsets[hi] if hi < n_games else fsize))

    all_r = []
    with Pool(args.workers) as pool:
        for cid, rs in pool.imap_unordered(check_chunk, tasks):
            all_r.extend(rs)
    n_hu = len(all_r)
    bad = [r for r in all_r if not r['ok']]
    print('games: %d, hu games: %d, fan match: %d, MISMATCH: %d (%.4f%%), %.1fs' % (
        n_games, n_hu, n_hu - len(bad), len(bad), 100.0 * len(bad) / max(n_hu, 1),
        time.time() - t0))
    by_mode = Counter(r['mode'] for r in bad)
    by_explained = Counter(r.get('explained_by', 'UNEXPLAINED') for r in bad)
    print('mismatch by mode:', dict(by_mode))
    print('mismatch by flag-toggle explanation:', dict(by_explained))
    for r in bad[:10]:
        print(' ', json.dumps(r, ensure_ascii=False, default=str)[:300])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'n_games': n_games, 'n_hu': n_hu, 'mismatches': bad}, f,
                  ensure_ascii=False, indent=1, default=str)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
