"""牌谱重放驱动:把 data.txt 格式的对局重放成对齐的 (obs, act) 样本。

逻辑与 baseline/SL/preprocess.py 完全一致(含 Ignore 多家宣告处理),
额外解析 Fan/Score/Huang 行用于 meta。被 preprocess_v1、replay_check、tests 共用。
"""


def iter_matches(path, limit=None):
    """逐局迭代,yield (match_id_str, lines)。lines 不含 Match 行自身。"""
    n = 0
    with open(path, encoding='UTF-8') as f:
        match_id = None
        lines = []
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == 'Match':
                if match_id is not None:
                    yield match_id, lines
                    n += 1
                    if limit is not None and n >= limit:
                        return
                match_id = t[1]
                lines = []
            else:
                lines.append(line)
        if match_id is not None:
            yield match_id, lines


def replay_match(match_id, lines, agent_clz, observer=None):
    """重放一局,返回:
    {
      'match_id': str,
      'obs': [seat][按时间顺序的 obs dict 列表](未过滤),
      'act': [seat][与 obs 对齐的动作 id],
      'scores': [4],         本局四家原始得分
      'winner': int,         胡牌者绝对座位,荒庄为 -1
      'fan': int,            番数,荒庄为 0
    }
    """
    obs = [[] for _ in range(4)]
    actions = [[] for _ in range(4)]
    agents = None
    curTile = None
    scores = [0, 0, 0, 0]
    winner = -1
    fan = 0

    for line in lines:
        t = line.split()
        if not t:
            continue
        if t[0] == 'Wind':
            agents = [agent_clz(i) for i in range(4)]
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
                actions[p].append(agents[p].response2action(' '.join(t[2:4])))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(' '.join(t[:4]))
                    else:
                        obs[i].append(agents[i].request2obs(' '.join(t[:4])))
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
                winner = p
            # Ignore 子句:同打多家宣告,被忽略者的动作也要记录
            if t[2] in ('Peng', 'Gang', 'Hu'):
                for k in range(5, 15, 5):
                    if len(t) > k:
                        q = int(t[k + 1])
                        a = t[k + 2].capitalize()  # 兼容 README 中的全大写写法
                        if a == 'Chi':
                            actions[q].pop()
                            actions[q].append(agents[q].response2action('Chi %s %s' % (curTile, t[k + 3])))
                        elif a == 'Peng':
                            actions[q].pop()
                            actions[q].append(agents[q].response2action('Peng %s' % t[k + 3]))
                        elif a == 'Gang':
                            actions[q].pop()
                            actions[q].append(agents[q].response2action('Gang %s' % t[k + 3]))
                        elif a == 'Hu':
                            actions[q].pop()
                            actions[q].append(agents[q].response2action('Hu'))
                        else:
                            raise ValueError('match %s: unknown ignore action %r' % (match_id, t[k + 2]))
                    else:
                        break
        elif t[0] == 'Fan':
            fan = int(t[1])
        elif t[0] == 'Score':
            scores = [int(x) for x in t[1:5]]
        elif t[0] == 'Huang':
            pass
        else:
            raise ValueError('match %s: unknown line %r' % (match_id, line.strip()))
        if observer is not None:
            # 每行处理完回调一次:t=本行 token,agents=四家真实状态(oracle 提取用),
            # obs=各座位已累积的决策点列表(回调方按长度增长检测新决策点)
            observer(t, agents, obs)

    for i in range(4):
        assert len(obs[i]) == len(actions[i]), \
            'match %s seat %d: obs/actions not matching!' % (match_id, i)
    return {
        'match_id': match_id,
        'obs': obs,
        'act': actions,
        'scores': scores,
        'winner': winner,
        'fan': fan,
    }


def check_legality(result):
    """对一局的所有决策点检查记录的动作是否在 action_mask 内。

    返回违规列表 [(seat, step, act_id, mask_sum), ...],目标为空。
    """
    violations = []
    for seat in range(4):
        for step, (o, a) in enumerate(zip(result['obs'][seat], result['act'][seat])):
            if not o['action_mask'][a]:
                violations.append((seat, step, int(a), int(o['action_mask'].sum())))
    return violations
