"""从重放数据验证相对座位映射:行动顺序的"下一个行动者"= (N+1)%4,
且 FeatureAgentV1 的相对索引 1(下家)/3(上家)与之一致。不凭直觉。"""
import os

from conftest import SAMPLE_TXT, DATA_TXT
from replay_driver import iter_matches
from feature_v1 import FeatureAgentV1


def _data_path(n_default):
    if os.path.exists(DATA_TXT):
        return DATA_TXT, int(os.environ.get('SEAT_GAMES', 2000))
    return SAMPLE_TXT, n_default


def test_next_actor_is_plus_one():
    """牌谱中:Play 后若无人宣告则下一行动者摸牌;Chi 只能由打牌者的逆时针下一家做。"""
    path, limit = _data_path(16)
    n_draw, n_chi = 0, 0
    for match_id, lines in iter_matches(path, limit=limit):
        events = []
        for line in lines:
            t = line.split()
            if t and t[0] == 'Player':
                events.append((int(t[1]), t[2]))
        for (p, a), (q, b) in zip(events, events[1:]):
            if a == 'Play' and b == 'Draw':
                assert q == (p + 1) % 4, \
                    'match %s: Play by %d followed by Draw by %d' % (match_id, p, q)
                n_draw += 1
            if a == 'Play' and b == 'Chi':
                assert q == (p + 1) % 4, \
                    'match %s: Play by %d followed by Chi by %d' % (match_id, p, q)
                n_chi += 1
    assert n_draw > 100 and n_chi > 10, 'not enough evidence (%d draws, %d chis)' % (n_draw, n_chi)


def test_agent_relative_mapping():
    """FeatureAgentV1:绝对座位 (seat+1)%4 的事件应落在相对索引 1(下家),(seat+3)%4 落在 3(上家)。"""
    for seat in range(4):
        agent = FeatureAgentV1(seat)
        agent.request2obs('Wind 0')
        agent.request2obs('Deal W1 W2 W3 T1 T2 T3 B1 B2 B3 F1 F1 F2 F3')
        nxt = (seat + 1) % 4
        agent.request2obs('Player %d Draw' % nxt)
        agent.request2obs('Player %d Play W5' % nxt)
        assert agent.history[1] == ['W5'], 'seat %d: next player should be relative 1' % seat
        prev = (seat + 3) % 4
        agent.request2obs('Player %d Draw' % prev)
        obs = agent.request2obs('Player %d Play W4' % prev)
        assert agent.history[3] == ['W4'], 'seat %d: previous player should be relative 3' % seat
        # 上家(相对 3)打出的牌才能吃:W4 可与手中 W2 W3 构成顺子
        chi_actions = [a for a in agent.valid
                       if FeatureAgentV1.OFFSET_ACT['Chi'] <= a < FeatureAgentV1.OFFSET_ACT['Peng']]
        assert chi_actions, 'seat %d: chi from upstream player must be available' % seat
        assert obs['action_mask'][chi_actions].all()
