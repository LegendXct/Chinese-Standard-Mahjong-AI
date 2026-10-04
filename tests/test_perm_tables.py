"""置换表结构正确性:双射、各动作区语义、行置换/逆置换闭合。"""
import numpy as np

from feature_v1 import FeatureAgentV1
from dataset_v1 import (SUIT_PERMS, TILE_PERMS, ACT_PERMS, MASK_GATHER, ROW_GATHER,
                        augment_sample)

OA = FeatureAgentV1.OFFSET_ACT
AGENT = FeatureAgentV1(0)


def _rename(resp, sigma):
    out = []
    for tok in resp.split():
        if len(tok) == 2 and tok[0] in 'WTB' and tok[1].isdigit():
            out.append('WTB'[sigma['WTB'.index(tok[0])]] + tok[1])
        else:
            out.append(tok)
    return ' '.join(out)


def test_bijections():
    for k in range(6):
        assert (np.sort(TILE_PERMS[k]) == np.arange(34)).all()
        assert (np.sort(ACT_PERMS[k]) == np.arange(235)).all()
        assert (ACT_PERMS[k][MASK_GATHER[k]] == np.arange(235)).all() or \
               (MASK_GATHER[k][ACT_PERMS[k]] == np.arange(235)).all()
    assert (ACT_PERMS[0] == np.arange(235)).all()
    assert (ROW_GATHER[0] == np.arange(4)).all()


def test_action_semantics_via_strings():
    """带牌面的动作(Play/Chi/AnGang/BuGang):置换后动作的 response 字符串
    应等于原 response 字符串做牌面改名。"""
    for k, sigma in enumerate(SUIT_PERMS):
        for a in range(235):
            resp = AGENT.action2response(a)
            resp_perm = AGENT.action2response(int(ACT_PERMS[k][a]))
            head = resp.split()[0]
            if head in ('Play', 'Chi', 'AnGang', 'BuGang') or \
                    (head == 'Gang' and len(resp.split()) > 1):
                assert resp_perm == _rename(resp, sigma), \
                    'k=%d a=%d: %r vs %r' % (k, a, resp, resp_perm)
            else:  # Pass/Hu/Peng/裸Gang:类型不变
                assert resp_perm.split()[0] == head


def test_tile_blocks():
    """Peng/Gang 区(response 不含牌面)按 34 牌索引置换。"""
    for k in range(6):
        for base in (OA['Peng'], OA['Gang']):
            for i in range(34):
                assert ACT_PERMS[k][base + i] == base + TILE_PERMS[k][i]


def test_round_trip_identity():
    """k 与其逆置换复合 = 恒等(obs/mask/act 全部还原)。"""
    rng = np.random.RandomState(0)
    obs = rng.randint(0, 255, (102, 4, 9)).astype(np.uint8)
    mask = rng.randint(0, 2, 235).astype(np.uint8)
    act = 50
    for k, sigma in enumerate(SUIT_PERMS):
        inv = [0, 0, 0]
        for s in range(3):
            inv[sigma[s]] = s
        k_inv = SUIT_PERMS.index(tuple(inv))
        o1, m1, a1 = augment_sample(obs, mask, act, k)
        o2, m2, a2 = augment_sample(o1, m1, a1, k_inv)
        assert (o2 == obs).all() and (m2 == mask).all() and a2 == act
