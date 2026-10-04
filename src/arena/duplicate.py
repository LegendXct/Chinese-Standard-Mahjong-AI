"""复式赛制:同一(牌墙+圈风)下候选策略在 4 个座位轮换。

基于 baseline 的 MahjongGBEnv(duplicate=True:每家从固定的 34 张墙段摸牌,
座位轮换时四手牌完全复现,即复式)。
"""
import os
import random
import sys

import numpy as np

_SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from feature_v1 import FeatureAgentV1  # noqa: E402  (亦把 baseline/RL 加入 path)
from env_fixed import MahjongGBEnvFixed as MahjongGBEnv  # noqa: E402


def make_wall(seed):
    tiles = []
    for _ in range(4):
        for i in range(1, 10):
            tiles.extend(['W%d' % i, 'B%d' % i, 'T%d' % i])
        for i in range(1, 5):
            tiles.append('F%d' % i)
        for i in range(1, 4):
            tiles.append('J%d' % i)
    random.Random(str(seed)).shuffle(tiles)
    return ' '.join(tiles)


def play_game(env, seat_policies, wall, wind, game_seed, decision_salt=0):
    """打一局。返回 {'scores': [4], 'winner': -1/seat, 'zimo': bool, 'dianpao': -1/seat, 'fan': int, 'steps': n}

    decision_salt: 0 时 reset 种子 = str((game_seed,i))(与老路径字节一致);非 0 时叠入
    salt → 改采样随机流但不改墙(argmax 不用 rng,故 salt 对 argmax 结果无影响)。
    """
    for i, pol in enumerate(seat_policies):
        seed = str((game_seed, i)) if not decision_salt else str((game_seed, i, decision_salt))
        pol.reset(seed=seed)
    obs = env.reset(prevalentWind=wind, tileWall=wall)
    done = False
    steps = 0
    rewards = {}
    while not done:
        action_dict = {}
        for name, o in obs.items():
            seat = int(name.split('_')[1]) - 1
            action_dict[name] = seat_policies[seat].act(o)
        obs, rewards, done = env.step(action_dict)
        steps += 1
        assert steps < 1000, 'game not terminating'
    scores = [rewards['player_%d' % (i + 1)] for i in range(4)]
    if all(s == 0 for s in scores):
        return {'scores': scores, 'winner': -1, 'zimo': False, 'dianpao': -1, 'fan': 0, 'steps': steps}
    # 非法动作时 env 给违规者 -30、其余 +10;合法胡牌赢家至少 +32,据此区分
    assert max(scores) >= 32, 'invalid action occurred: %s' % scores
    winner = int(np.argmax(scores))
    losers = sorted(s for i, s in enumerate(scores) if i != winner)
    zimo = losers[0] == losers[-1]
    if zimo:
        fan = scores[winner] // 3 - 8
        dianpao = -1
    else:
        fan = scores[winner] - 24
        dianpao = int(np.argmin(scores))
    return {'scores': scores, 'winner': winner, 'zimo': zimo, 'dianpao': dianpao,
            'fan': fan, 'steps': steps}


def play_group(group_id, base_seed, cand_policy, opp_policy, mode='1v3',
               opp_policies=None, decision_salt=0):
    """一组复式:同墙同圈风,候选轮换座位。返回每局记录列表(含候选视角指标)。

    opp_policies: 可选(仅 1v3)。给"非候选三座位"各一个独立对手对象(异质对手 Test#3a,
                  或同质但独立采样 rng 的 clone Test#1);None 时沿用单一 opp_policy(老路径,
                  三座位共享同一对象,argmax 下等价)。
    decision_salt: 透传 play_game,同墙换采样随机流复测用。
    二者均默认关闭 → 与老 argmax 路径字节级一致(Step0 的 0.000/+4.42 照常复现)。
    """
    wall = make_wall((base_seed, group_id, 'wall'))
    wind = random.Random(str((base_seed, group_id, 'wind'))).randrange(4)
    env = MahjongGBEnv(config={'agent_clz': FeatureAgentV1})
    if mode == '1v3':
        arrangements = [[1 if i == s else 0 for i in range(4)] for s in range(4)]
    elif mode == '2v2':
        arrangements = [[1, 0, 1, 0], [0, 1, 0, 1]]
    else:
        raise ValueError(mode)
    if opp_policies is not None and mode != '1v3':
        raise ValueError('opp_policies 仅支持 1v3')
    records = []
    for rot, arr in enumerate(arrangements):
        if opp_policies is None:
            seat_policies = [cand_policy if flag else opp_policy for flag in arr]
        else:
            seat_policies, oi = [], 0
            for flag in arr:
                if flag:
                    seat_policies.append(cand_policy)
                else:
                    seat_policies.append(opp_policies[oi])
                    oi += 1
        r = play_game(env, seat_policies, wall, wind,
                      game_seed=(base_seed, group_id, rot), decision_salt=decision_salt)
        cand_seats = [i for i, f in enumerate(arr) if f]
        for cs in cand_seats:
            records.append({
                'group': group_id, 'rot': rot, 'cand_seat': cs,
                'score': r['scores'][cs],
                'others_mean': (sum(r['scores']) - r['scores'][cs]) / 3.0,
                'hu': r['winner'] == cs,
                'zimo': r['winner'] == cs and r['zimo'],
                'dianpao': r['dianpao'] == cs,
                'fan': r['fan'] if r['winner'] == cs else 0,
                'huang': r['winner'] == -1,
            })
    return records
