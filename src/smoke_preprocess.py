"""冒烟测试 A:对 sample.txt(16 局)跑通 baseline 的数据预处理逻辑。

逻辑照搬 baseline/SL/preprocess.py,仅参数化输入/输出路径并加计时,
不改动 baseline 原文件。在服务器项目根目录运行:
    python src/smoke_preprocess.py
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'baseline', 'SL'))
from feature import FeatureAgent  # noqa: E402

INPUT = os.path.join(ROOT, 'baseline', 'SL', 'data', 'sample.txt')
OUTDIR = os.path.join(ROOT, 'tmp', 'preprocess_out')
os.makedirs(OUTDIR, exist_ok=True)

obs = [[] for i in range(4)]
actions = [[] for i in range(4)]
matchid = -1
l = []
shapes_reported = False


def filterData():
    global obs
    global actions
    newobs = [[] for i in range(4)]
    newactions = [[] for i in range(4)]
    for i in range(4):
        for j, o in enumerate(obs[i]):
            if o['action_mask'].sum() > 1:  # ignore states with single valid action (Pass)
                newobs[i].append(o)
                newactions[i].append(actions[i][j])
    obs = newobs
    actions = newactions


def saveData():
    global shapes_reported
    assert [len(x) for x in obs] == [len(x) for x in actions], 'obs actions not matching!'
    l.append(sum([len(x) for x in obs]))
    arr_obs = np.stack([x['observation'] for i in range(4) for x in obs[i]]).astype(np.int8)
    arr_mask = np.stack([x['action_mask'] for i in range(4) for x in obs[i]]).astype(np.int8)
    arr_act = np.array([x for i in range(4) for x in actions[i]])
    np.savez(os.path.join(OUTDIR, '%d.npz' % matchid), obs=arr_obs, mask=arr_mask, act=arr_act)
    if not shapes_reported:
        print('first match arrays: obs%s mask%s act%s (dtype %s/%s/%s)' % (
            arr_obs.shape, arr_mask.shape, arr_act.shape,
            arr_obs.dtype, arr_mask.dtype, arr_act.dtype))
        shapes_reported = True
    for x in obs:
        x.clear()
    for x in actions:
        x.clear()


t_start = time.time()
with open(INPUT, encoding='UTF-8') as f:
    line = f.readline()
    while line:
        t = line.split()
        if len(t) == 0:
            line = f.readline()
            continue
        if t[0] == 'Match':
            agents = [FeatureAgent(i) for i in range(4)]
            matchid += 1
        elif t[0] == 'Wind':
            for agent in agents:
                agent.request2obs(line)
        elif t[0] == 'Player':
            p = int(t[1])
            if t[2] == 'Deal':
                agents[p].request2obs(' '.join(t[2:]))
            elif t[2] == 'Draw':
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs(' '.join(t[2:])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs(' '.join(t[:3]))
            elif t[2] == 'Play':
                actions[p].pop()
                actions[p].append(agents[p].response2action(' '.join(t[2:])))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(line)
                    else:
                        obs[i].append(agents[i].request2obs(line))
                        actions[i].append(0)
                curTile = t[3]
            elif t[2] == 'Chi':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[3])))
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs('Player %d Chi %s' % (p, t[3])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
            elif t[2] == 'Peng':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Peng %s' % t[3]))
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs('Player %d Peng %s' % (p, t[3])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
            elif t[2] == 'Gang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Gang %s' % t[3]))
                for i in range(4):
                    agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
            elif t[2] == 'AnGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('AnGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                    else:
                        agents[i].request2obs('Player %d AnGang' % p)
            elif t[2] == 'BuGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('BuGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d BuGang %s' % (p, t[3]))
                    else:
                        obs[i].append(agents[i].request2obs('Player %d BuGang %s' % (p, t[3])))
                        actions[i].append(0)
            elif t[2] == 'Hu':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Hu'))
            # Deal with Ignore clause
            if t[2] in ['Peng', 'Gang', 'Hu']:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p = int(t[k + 1])
                        if t[k + 2] == 'Chi':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[k + 3])))
                        elif t[k + 2] == 'Peng':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Peng %s' % t[k + 3]))
                        elif t[k + 2] == 'Gang':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Gang %s' % t[k + 3]))
                        elif t[k + 2] == 'Hu':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Hu'))
                    else:
                        break
        elif t[0] == 'Score':
            filterData()
            saveData()
        line = f.readline()

elapsed = time.time() - t_start
n_matches = matchid + 1
with open(os.path.join(OUTDIR, 'count.json'), 'w') as f:
    json.dump(l, f)
print('matches: %d, total samples: %d' % (n_matches, sum(l)))
print('total time: %.2fs, per match: %.3fs' % (elapsed, elapsed / n_matches))
print('estimated full 98209 matches: %.1f min' % (elapsed / n_matches * 98209 / 60))
