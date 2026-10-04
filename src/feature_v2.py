"""特征 v2:在 v1 的 102×4×9 空间张量之上增加一支"标量向量" vec。

设计要点(关键决策):
- **空间张量与 v1 逐元素相同**(feature_v2 不改 _obs),故数据管线可直接复用 v1 的
  obs.dat/mask.dat/act.dat,只另存 vec.dat(省 21.5GB 磁盘)。
- **vec 是 obs 的确定性函数**:compute_vec_from_obs(obs) 单一真源——训练建表、arena
  评测、线上部署全调它,杜绝 live-state 与建表口径不一致的隐患;arena 评 v2 也无需
  改环境(候选策略自己从 obs 现算 vec)。
- vec 主体是 Arthals 标杆里贡献最大的"每弃牌质量"先验:对 34 种弃牌各给
  (弃后向听/6, 弃后有效进张未见张数/34, 弃后是否听牌),仅出牌态(手牌数%3==2)有效,
  否则为默认(向听=1.0 最差、其余 0)。附少量 obs 可解码的全局标量。
  注:其他家暗手牌数不在自视角 obs 中,本版不含(Arthals 实测此类全局特征边界收益小,
  且强信号在每弃牌块);若后续需要可改 re-replay 取 live 态。
"""
import numpy as np

import feature_v1 as fv1
from feature_v1 import shanten_useful_wait  # 与 v1 向听通道同一函数,缓存复用

FEATURE_VERSION = "v2"

_TILE_LIST = fv1.FeatureAgentV1.TILE_LIST
_OFFSET_TILE = fv1.FeatureAgentV1.OFFSET_TILE

# vec 布局
PD_SHANTEN = 0    # 34  每弃牌:弃后向听 / 6(默认 1.0)
PD_USEFUL = 34    # 34  每弃牌:弃后有效进张未见张数 / 34(默认 0)
PD_TENPAI = 68    # 34  每弃牌:弃后是否听牌 0/1(默认 0)
G_CUR_SHANTEN = 102   # 当前(未弃)向听 / 6
G_USEFUL_UNSEEN = 103  # 当前有效进张未见总数(取 obs CH_USEFUL_UNSEEN,已 /255)
G_WALL_TOTAL = 104     # 牌墙总剩余(取 obs CH_WALL_TOTAL,已 /255)
G_WALL_SELF = 105      # 自己剩余可摸(取 obs CH_WALL_SELF,已 /255)
G_HAND_CNT = 106       # 自己手牌数 / 14
VEC_DIM = 107

# 每弃牌块在花色置换下需按"弃牌牌种"重排(全局标量花色无关)
PD_BLOCKS = (PD_SHANTEN, PD_USEFUL, PD_TENPAI)


def _decode_counts(obs, base):
    """obs (102,4,9):channels [base,base+4) 的温度计求和 -> 34 维计数。"""
    plane = obs[base:base + 4].sum(axis=0).reshape(36)[:34]
    return plane.astype(np.int32)


def compute_vec_from_obs(obs):
    """obs: (102,4,9) uint8(未归一化)。返回 float32 (VEC_DIM,)。"""
    vec = np.zeros(VEC_DIM, dtype=np.float32)
    vec[PD_SHANTEN:PD_SHANTEN + 34] = 1.0   # 默认最差向听

    hand_cnt = _decode_counts(obs, fv1.CH_HAND)
    unseen = _decode_counts(obs, fv1.CH_UNSEEN)
    total = int(hand_cnt.sum())

    # 全局标量(均 obs 可解码;×255 通道取任一格,已是 /255 的值再 /255)
    sh_oh = obs[fv1.CH_SHANTEN:fv1.CH_SHANTEN + 7, 0, 0]
    cur_sh = int(np.argmax(sh_oh)) if sh_oh.any() else 6
    vec[G_CUR_SHANTEN] = min(cur_sh, 6) / 6.0
    vec[G_USEFUL_UNSEEN] = obs[fv1.CH_USEFUL_UNSEEN, 0, 0] / 255.0
    vec[G_WALL_TOTAL] = obs[fv1.CH_WALL_TOTAL, 0, 0] / 255.0
    vec[G_WALL_SELF] = obs[fv1.CH_WALL_SELF, 0, 0] / 255.0
    vec[G_HAND_CNT] = total / 14.0

    if total % 3 != 2:        # 非出牌态:每弃牌块保持默认
        return vec
    hand = []
    for i in range(34):
        hand.extend([_TILE_LIST[i]] * int(hand_cnt[i]))
    for i in range(34):
        if hand_cnt[i] == 0:
            continue
        rest = list(hand)
        rest.remove(_TILE_LIST[i])
        try:
            sh, useful = shanten_useful_wait(tuple(sorted(rest)))
        except Exception:
            sh, useful = 6, ()
        vec[PD_SHANTEN + i] = min(sh, 6) / 6.0
        uu = 0
        for t in useful:
            uu += int(unseen[_OFFSET_TILE[t]])
        vec[PD_USEFUL + i] = min(uu, 34) / 34.0
        vec[PD_TENPAI + i] = 1.0 if sh <= 0 else 0.0
    return vec


class FeatureAgentV2(fv1.FeatureAgentV1):
    """线上/RL 用:_obs() 沿用 v1,额外提供 vec(从 obs 现算,与建表同源)。"""

    def obs_with_vec(self):
        o = self._obs()
        o['vec'] = compute_vec_from_obs(o['observation'])
        return o
