"""特征 v2.1:在 v2 的 107 维 vec 之上再拼 10 维"番种/牌型先验"(EXT)。

动机(gen2 阶段一-C):vec(每弃牌标准向听)拿了 +2.3pp;参照 Arthals VEC-117 的特征丰度,
试补一组 **CNN 难自算、且与"做大牌"对齐** 的标量,看能否再榨 +1~2pp(以 arena 棋力为准)。

设计纪律(与 v2 一致):
- **vec 仍是 obs 的确定性函数**(compute_vec_ext_from_obs),建表/arena/部署单一真源。
- EXT 全部 **花色无关**(max-over-suit / 计数 / 门清七对),故花色置换增广下 **EXT 段恒等**,
  避免逐花色重排的易错点;前 107 维沿用 v2 的牌种置换(vec_gather_ext)。
- 全部仅从 **自视角已解码量**(暗手牌计数 + 未见 + 由暗手牌总数推出的副露数)计算,
  不解码副露通道(避免吃/碰/杠编码歧义);七对仅门清有效(有副露即判不可行)。
- 数值归一到 ~[0,1],与 base vec 同尺度。
"""
import numpy as np

import feature_v1 as fv1
import feature_v2 as fv2
from feature_v2 import compute_vec_from_obs, _decode_counts

FEATURE_VERSION = "v2.1"

BASE_DIM = fv2.VEC_DIM          # 107
EXT_DIM = 10
VEC_DIM_EXT = BASE_DIM + EXT_DIM  # 117

# EXT 段索引(接在 base 之后)
_E = BASE_DIM
EXT_QIDUI_SHANTEN = _E + 0   # 七对向听/6(门清才有效,否则 1.0 表不可行)
EXT_QIDUI_USEFUL = _E + 1    # 七对进张未见(单张待成对的未见数)/34
EXT_FLUSH_FRAC = _E + 2      # 暗手牌最大同数字花色占比(清一色潜力)
EXT_HONOR_FRAC = _E + 3      # 字牌数/14(字一色/混一色相关)
EXT_PAIR_CNT = _E + 4        # 对子数/7
EXT_TRIPLET_CNT = _E + 5     # 暗刻数/4
EXT_NUM_MELDS = _E + 6       # 副露数/4(由暗手牌总数推出)
EXT_KINDS = _E + 7           # 不同牌种数/14(越小越偏碰碰/清一)
EXT_MAX_TILE = _E + 8        # 单种最大张数/4(=4 提示可杠)
EXT_HUNFLUSH_FRAC = _E + 9   # (最大花色+字牌)占比(混一色潜力)

# W:0-8 T:9-17 B:18-26(三数字花色),F+J:27-33(字牌)
_SUITS = ((0, 9), (9, 18), (18, 27))
_HONOR = tuple(range(27, 34))


def _ext_features(hand_cnt, unseen, total):
    out = np.zeros(EXT_DIM, dtype=np.float32)
    num_melds = ((13 - total) // 3) if total % 3 == 1 else ((14 - total) // 3)
    num_melds = max(0, min(4, num_melds))
    kinds = int((hand_cnt >= 1).sum())
    pairs = int((hand_cnt == 2).sum())
    trips = int((hand_cnt >= 3).sum())

    if num_melds == 0:                       # 七对仅门清成立
        npair = int((hand_cnt >= 2).sum())
        sh7 = 6 - npair + max(0, 7 - kinds)
        out[0] = min(max(sh7, 0), 6) / 6.0
        uu = sum(int(unseen[i]) for i in range(34) if hand_cnt[i] == 1)
        out[1] = min(uu, 34) / 34.0
    else:
        out[0] = 1.0

    suit_sums = [int(hand_cnt[a:b].sum()) for a, b in _SUITS]
    honor = sum(int(hand_cnt[i]) for i in _HONOR)
    tt = max(total, 1)
    out[2] = max(suit_sums) / tt
    out[3] = honor / 14.0
    out[4] = pairs / 7.0
    out[5] = trips / 4.0
    out[6] = num_melds / 4.0
    out[7] = kinds / 14.0
    out[8] = int(hand_cnt.max()) / 4.0
    out[9] = (max(suit_sums) + honor) / tt
    return out


def compute_vec_ext_from_obs(obs):
    """obs: (102,4,9) uint8。返回 float32 (VEC_DIM_EXT,) = [base 107 | ext 10]。"""
    base = compute_vec_from_obs(obs)
    hand_cnt = _decode_counts(obs, fv1.CH_HAND)
    unseen = _decode_counts(obs, fv1.CH_UNSEEN)
    total = int(hand_cnt.sum())
    ext = _ext_features(hand_cnt, unseen, total)
    return np.concatenate([base, ext]).astype(np.float32)


def vec_gather_ext():
    """(6, VEC_DIM_EXT) 花色置换 gather:前 107 维按牌种重排(同 dataset_v2.VEC_GATHER),
    EXT 10 维花色无关恒等。独立构造(不 import dataset_v2,避免循环依赖)。"""
    from dataset_v1 import TILE_PERMS
    tile_gather = np.stack([np.argsort(p) for p in TILE_PERMS])
    n = len(TILE_PERMS)
    g = np.tile(np.arange(VEC_DIM_EXT), (n, 1))
    for k in range(n):
        for blk in fv2.PD_BLOCKS:
            g[k, blk:blk + 34] = blk + tile_gather[k]
    return g


class FeatureAgentV2Ext(fv1.FeatureAgentV1):
    """线上/RL 用:_obs() 沿用 v1,额外提供扩展 vec(从 obs 现算,与建表同源)。"""

    def obs_with_vec(self):
        o = self._obs()
        o['vec'] = compute_vec_ext_from_obs(o['observation'])
        return o
