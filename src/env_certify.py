"""3b env 准入认证:从牌谱重建四家 34 张墙段,把全部 98209 局注入 env_fixed
逐步重放,断言全程无非法判定、终局四家得分与 Score 行完全一致。
这是 MahjongGBEnvFixed 进入 M3 自对弈训练的准入测试。

墙重建依据 baseline env 的消耗顺序(env.reset -> _deal 每家 13 张、_draw 摸牌,
均为 tileWall[seat].pop() 从段尾弹出):
  segment_i = 填充牌 + reversed(起手13张 + 该家按序全部 Draw)
填充牌取全局未出现牌的多重集,任意顺序(永远不会被摸到)。

用法:
  python src/env_certify.py --data baseline/SL/data/sample.txt --workers 4
  python src/env_certify.py --data baseline/SL/data/data.txt --workers 96
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
from feature_v1 import FeatureAgentV1  # noqa: E402
from env_fixed import MahjongGBEnvFixed  # noqa: E402
from preprocess_v1 import index_match_offsets, _iter_matches_in_slice  # noqa: E402

FULL_SET = []
for _ in range(4):
    for i in range(1, 10):
        FULL_SET.extend(['W%d' % i, 'T%d' % i, 'B%d' % i])
    for i in range(1, 5):
        FULL_SET.append('F%d' % i)
    for i in range(1, 4):
        FULL_SET.append('J%d' % i)


class CertifyError(Exception):
    pass


def parse_match(lines):
    wind = None
    deals = [None] * 4
    events = []
    scores = None
    for line in lines:
        t = line.split()
        if not t:
            continue
        if t[0] == 'Wind':
            wind = int(t[1])
        elif t[0] == 'Player' and t[2] == 'Deal':
            deals[int(t[1])] = t[3:]
        elif t[0] == 'Score':
            scores = [int(x) for x in t[1:5]]
        elif t[0] in ('Fan', 'Huang'):
            pass
        else:
            events.append(t)
    return wind, deals, events, scores


def build_wall(deals, events):
    consumed = [list(d) for d in deals]
    for t in events:
        if t[2] == 'Draw':
            consumed[int(t[1])].append(t[3])
    used = Counter()
    for c in consumed:
        if len(c) > 34:
            raise CertifyError('seat consumed %d > 34' % len(c))
        used.update(c)
    remainder = list((Counter(FULL_SET) - used).elements())
    if len(remainder) != 136 - sum(len(c) for c in consumed):
        raise CertifyError('tile multiset inconsistent')
    segs = []
    pos = 0
    for i in range(4):
        n_fill = 34 - len(consumed[i])
        segs.append(remainder[pos:pos + n_fill] + list(reversed(consumed[i])))
        pos += n_fill
    return ' '.join(segs[0] + segs[1] + segs[2] + segs[3])


def certify_match(match_id, lines):
    """返回 None(通过)或失败原因 dict。"""
    wind, deals, events, logged_scores = parse_match(lines)
    wall = build_wall(deals, events)
    env = MahjongGBEnvFixed(config={'agent_clz': FeatureAgentV1})
    obs = env.reset(prevalentWind=wind, tileWall=wall)
    for i in range(4):
        if Counter(env.hands[i]) != Counter(deals[i]):
            raise CertifyError('deal mismatch seat %d' % i)

    ei = 0  # 事件游标

    def next_ev():
        nonlocal ei
        if ei >= len(events):
            raise CertifyError('log exhausted while env still running')
        t = events[ei]
        ei += 1
        return t

    def peek_ev():
        return events[ei] if ei < len(events) else None

    def consume_draw():
        t = next_ev()
        if t[2] != 'Draw':
            raise CertifyError('expected Draw, got %s' % ' '.join(t))
        p = int(t[1])
        if env.curPlayer != p or env.curTile != t[3]:
            raise CertifyError('draw mismatch: env P%d %s vs log P%d %s'
                               % (env.curPlayer, env.curTile, p, t[3]))

    A = env.agents
    names = env.agent_names
    consume_draw()  # 起手后第一摸
    done = False
    rewards = {}
    steps = 0
    while not done:
        steps += 1
        if steps > 600:
            raise CertifyError('not terminating')
        action_dict = {}
        state = env.state
        if state in (0, 1):
            p = env.curPlayer
            t = next_ev()
            if int(t[1]) != p:
                raise CertifyError('expected actor P%d, got %s' % (p, ' '.join(t)))
            ev = t[2]
            if ev == 'Play':
                resp = 'Play %s' % t[3]
            elif ev == 'Hu':
                resp = 'Hu'
            elif ev == 'AnGang':
                resp = 'AnGang %s' % t[3]
            elif ev == 'BuGang':
                resp = 'BuGang %s' % t[3]
            else:
                raise CertifyError('unexpected event in state %d: %s' % (state, ' '.join(t)))
            action_dict[names[p]] = A[p].response2action(resp)
        elif state == 2:
            for i in range(4):
                if i != env.curPlayer:
                    action_dict[names[i]] = 0
            t = peek_ev()
            pending_draw = t is not None and t[2] == 'Draw'
            if t is not None and t[2] in ('Chi', 'Peng', 'Gang', 'Hu'):
                t = next_ev()
                q = int(t[1])
                if t[2] == 'Chi':
                    resp = 'Chi %s %s' % (env.curTile, t[3])
                elif t[2] == 'Peng':
                    resp = 'Peng %s' % t[3]
                elif t[2] == 'Gang':
                    resp = 'Gang %s' % t[3]
                else:
                    resp = 'Hu'
                action_dict[names[q]] = A[q].response2action(resp)
                # Ignore 子句:被忽略的宣告也注入,验证 env 的优先级裁决与裁判一致
                for k in range(4, len(t) - 1, 5):
                    if t[k] != 'Ignore':
                        break
                    q2 = int(t[k + 2])
                    a2 = t[k + 3].capitalize()
                    if a2 == 'Chi':
                        r2 = 'Chi %s %s' % (env.curTile, t[k + 4])
                    elif a2 == 'Peng':
                        r2 = 'Peng %s' % t[k + 4]
                    elif a2 == 'Gang':
                        r2 = 'Gang %s' % t[k + 4]
                    else:
                        r2 = 'Hu'
                    action_dict[names[q2]] = A[q2].response2action(r2)
        elif state == 3:
            for i in range(4):
                if i != env.curPlayer:
                    action_dict[names[i]] = 0
            t = peek_ev()
            if t is not None and t[2] == 'Hu':
                t = next_ev()
                action_dict[names[int(t[1])]] = A[int(t[1])].response2action('Hu')
                for k in range(4, len(t) - 1, 5):
                    if t[k] != 'Ignore':
                        break
                    if t[k + 3].capitalize() == 'Hu':
                        action_dict[names[int(t[k + 2])]] = \
                            A[int(t[k + 2])].response2action('Hu')
        obs, rewards, done = env.step(action_dict)
        if not done and env.state == 1:
            consume_draw()
    if min(rewards.values()) == -30 and max(rewards.values()) == 10:
        raise CertifyError('env judged INVALID action')
    env_scores = [rewards[names[i]] for i in range(4)]
    if env_scores != logged_scores:
        raise CertifyError('score mismatch: env %s vs log %s' % (env_scores, logged_scores))
    if ei < len(events):
        raise CertifyError('log has %d unconsumed events' % (len(events) - ei))
    return None


def certify_chunk(task):
    chunk_id, path, start, end = task
    n_ok = 0
    failures = []
    no_op = lambda self: {'observation': None, 'action_mask': None}
    orig = FeatureAgentV1._obs
    FeatureAgentV1._obs = no_op  # env 只要 reward/裁决,跳过特征渲染省向听计算
    try:
        for match_id, lines in _iter_matches_in_slice(path, start, end):
            try:
                certify_match(match_id, lines)
                n_ok += 1
            except CertifyError as e:
                failures.append({'match_id': match_id, 'reason': str(e)})
            except Exception as e:
                failures.append({'match_id': match_id, 'reason': 'EXC: %r' % e})
    finally:
        FeatureAgentV1._obs = orig
    return chunk_id, n_ok, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--out', default='results/env_certify.json')
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

    n_ok = 0
    failures = []
    with Pool(args.workers) as pool:
        for cid, ok, fails in pool.imap_unordered(certify_chunk, tasks):
            n_ok += ok
            failures.extend(fails)
    print('games: %d, certified: %d, FAILURES: %d, %.1fs' % (
        n_games, n_ok, len(failures), time.time() - t0))
    for f in failures[:10]:
        print(' ', f)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'n_games': n_games, 'n_ok': n_ok, 'failures': failures}, f,
                  ensure_ascii=False, indent=1)
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
