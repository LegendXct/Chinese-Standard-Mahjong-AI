"""统一策略接口与三种实现:Random / RuleBot(向听贪心)/ NN。

Policy.act(obs_dict) -> action_id,obs_dict 即 FeatureAgentV1._obs() 的输出
({'observation': uint8 (C,4,9), 'action_mask': uint8 (235,)})。
RuleBot 只从观测平面解码(手牌/未见数/待响应牌),同时间接检验特征正确性。
"""
import os
import random
import sys

import numpy as np

_SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import feature_v1 as fv1  # noqa: E402
from feature_v1 import shanten_useful_wait, shanten_useful_discard  # noqa: E402

OA = fv1.FeatureAgentV1.OFFSET_ACT
TILES = fv1.FeatureAgentV1.TILE_LIST


class Policy:
    def reset(self, seed=None):
        pass

    def act(self, obs):
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)

    def act(self, obs):
        return int(self.rng.choice(np.flatnonzero(obs['action_mask'])))


def _counts(planes):
    """(4,4,9) 温度计 -> 34 维计数。"""
    return planes.reshape(4, 36).sum(axis=0)[:34].astype(np.int32)


def _tiles_from_counts(cnt):
    out = []
    for i, c in enumerate(cnt):
        out.extend([TILES[i]] * int(c))
    return tuple(out)


def _useful_unseen(useful, unseen):
    return sum(int(unseen[fv1.FeatureAgentV1.OFFSET_TILE[t]]) for t in useful)


class RuleBotPolicy(Policy):
    """向听数贪心(评测锚点,确定性):
    - 能胡必胡;
    - 出牌:使(向听数, -有效进张未见数)最小的弃牌;
    - 暗杠/补杠/明杠:杠后向听不升即做;
    - 吃/碰:仅当严格降低向听数。
    """

    def act(self, obs):
        mask = obs['action_mask']
        if mask[OA['Hu']]:
            return OA['Hu']
        o = np.asarray(obs['observation']).reshape(fv1.NUM_CHANNELS, 36)
        hand_cnt = _counts(o[fv1.CH_HAND:fv1.CH_HAND + 4].reshape(4, 4, 9))
        unseen = _counts(o[fv1.CH_UNSEEN:fv1.CH_UNSEEN + 4].reshape(4, 4, 9))
        legal = np.flatnonzero(mask)
        plays = [a for a in legal if OA['Play'] <= a < OA['Chi']]

        if plays:
            # 出牌类决策(可能附带暗杠/补杠选项)
            best_a, best_key = None, None
            for a in plays:
                cnt = hand_cnt.copy()
                cnt[a - OA['Play']] -= 1
                s, u = shanten_useful_wait(_tiles_from_counts(cnt))
                key = (s, -_useful_unseen(u, unseen), a)
                if best_key is None or key < best_key:
                    best_a, best_key = a, key
            for a in legal:
                if OA['AnGang'] <= a < OA['BuGang']:
                    take, ti = 4, a - OA['AnGang']
                elif a >= OA['BuGang']:
                    take, ti = 1, a - OA['BuGang']
                else:
                    continue
                cnt = hand_cnt.copy()
                cnt[ti] -= take
                s, u = shanten_useful_wait(_tiles_from_counts(cnt))
                if s <= best_key[0]:  # 杠后向听不升即杠(多一次摸牌且增番)
                    return int(a)
            return int(best_a)

        if mask[OA['Pass']]:
            # 响应类决策:吃/碰/明杠/过(抢杠的胡在开头处理过,剩下只能过)
            cur = int(np.argmax(o[fv1.CH_CUR_TILE][:34])) if o[fv1.CH_CUR_TILE].any() else -1
            s_now, u_now = shanten_useful_wait(_tiles_from_counts(hand_cnt))
            qualified = []
            for a in legal:
                if OA['Chi'] <= a < OA['Peng']:
                    t = a - OA['Chi']
                    center = (t // 21) * 9 + (t % 21) // 3 + 1
                    cnt = hand_cnt.copy()
                    for ti in (center - 1, center, center + 1):
                        cnt[ti] -= 1
                    cnt[cur] += 1  # 被吃的牌来自桌面,不消耗手牌
                    s, u = shanten_useful_discard(_tiles_from_counts(cnt))
                    if s < s_now:
                        qualified.append(((s, -_useful_unseen(u, unseen), a), a))
                elif OA['Peng'] <= a < OA['Gang']:
                    cnt = hand_cnt.copy()
                    cnt[a - OA['Peng']] -= 2
                    s, u = shanten_useful_discard(_tiles_from_counts(cnt))
                    if s < s_now:
                        qualified.append(((s, -_useful_unseen(u, unseen), a), a))
                elif OA['Gang'] <= a < OA['AnGang']:
                    cnt = hand_cnt.copy()
                    cnt[a - OA['Gang']] -= 3
                    s, u = shanten_useful_wait(_tiles_from_counts(cnt))
                    if s <= s_now:
                        qualified.append(((s, -_useful_unseen(u, unseen), a), a))
            if qualified:
                return int(min(qualified)[1])
            return OA['Pass']

        # 不应到达:除终局外总有合法动作
        return int(legal[0])


class NNPolicy(Policy):
    """CPU 推理。model 需接受 {'observation','action_mask'} 并返回 (logits, value),
    logits 已做 mask(同 baseline CNNModel 约定)。temperature=0 为 argmax。
    claim_tenpai=True 时叠加"抢牌成型"启发式:响应态模型选 PASS、但某合法
    吃/碰/杠能把向听严格降到 ≤1(听牌/将听),则改取该最优鸣牌(针对天梯
    统计诊断的"做牌慢、胡率低";风险是暴露增多,A/B 验证)。"""

    def __init__(self, model, checkpoint=None, temperature=0.0, claim_tenpai=False):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        self.model = model
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        self.model.eval()
        self.temperature = temperature
        self.claim_tenpai = claim_tenpai
        self.rng = random.Random(0)

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)

    def _claim_override(self, obs, base_action):
        """base_action==Pass 且响应态时,寻找能降向听到 ≤1 的最优鸣牌。"""
        mask = obs['action_mask']
        if base_action != OA['Pass'] or not mask[OA['Pass']]:
            return base_action
        o = np.asarray(obs['observation']).reshape(fv1.NUM_CHANNELS, 36)
        hand_cnt = _counts(o[fv1.CH_HAND:fv1.CH_HAND + 4].reshape(4, 4, 9))
        unseen = _counts(o[fv1.CH_UNSEEN:fv1.CH_UNSEEN + 4].reshape(4, 4, 9))
        cur = int(np.argmax(o[fv1.CH_CUR_TILE][:34])) if o[fv1.CH_CUR_TILE].any() else -1
        s_now, _ = shanten_useful_wait(_tiles_from_counts(hand_cnt))
        best = None
        for a in np.flatnonzero(mask):
            if OA['Chi'] <= a < OA['Peng']:
                t = a - OA['Chi']
                center = (t // 21) * 9 + (t % 21) // 3 + 1
                cnt = hand_cnt.copy()
                for ti in (center - 1, center, center + 1):
                    cnt[ti] -= 1
                cnt[cur] += 1
                s, u = shanten_useful_discard(_tiles_from_counts(cnt))
            elif OA['Peng'] <= a < OA['Gang']:
                cnt = hand_cnt.copy()
                cnt[a - OA['Peng']] -= 2
                s, u = shanten_useful_discard(_tiles_from_counts(cnt))
            elif OA['Gang'] <= a < OA['AnGang']:
                cnt = hand_cnt.copy()
                cnt[a - OA['Gang']] -= 3
                s, u = shanten_useful_wait(_tiles_from_counts(cnt))
            else:
                continue
            if s <= 1 and s < s_now:    # 严格降向听且到听/将听
                key = (s, -_useful_unseen(u, unseen), int(a))
                if best is None or key < best:
                    best = key
        return best[2] if best else base_action

    def act(self, obs):
        torch = self.torch
        from dataset_v1 import normalize_obs
        with torch.no_grad():
            ot = normalize_obs(torch.from_numpy(np.ascontiguousarray(obs['observation'])).unsqueeze(0))
            mt = torch.from_numpy(np.ascontiguousarray(obs['action_mask'])).float().unsqueeze(0)
            logits, _ = self.model({'observation': ot, 'action_mask': mt})
            logits = logits[0]
            if self.temperature <= 0:
                action = int(logits.argmax())
            else:
                probs = torch.softmax(logits / self.temperature, dim=0).numpy()
                probs = probs * np.asarray(obs['action_mask'])
                probs /= probs.sum()
                action = int(self.rng.choices(range(235), weights=probs)[0])
        if self.claim_tenpai:
            action = self._claim_override(obs, action)
        return action

    def clone_shared(self):
        """共享底层 torch model、独立采样 rng 的副本(多座位独立决策种子用)。"""
        c = NNPolicy.__new__(NNPolicy)
        c.torch = self.torch
        c.model = self.model            # 共享(eval/no_grad 推理无状态)
        c.temperature = self.temperature
        c.claim_tenpai = self.claim_tenpai
        c.rng = random.Random(0)        # 独立 rng
        return c


class NNPolicyV2(Policy):
    """gen2 双输入模型(obs+vec)CPU 推理。vec 由 feature_v2.compute_vec_from_obs
    从 obs 现算(与训练建表同源),故 arena 环境无需改动。temperature=0 为 argmax。"""

    def __init__(self, model, temperature=0.0, vecfn=None):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        self.model = model
        self.model.eval()
        self.temperature = temperature
        self.rng = random.Random(0)
        if vecfn is None:
            from feature_v2 import compute_vec_from_obs
            vecfn = compute_vec_from_obs
        self._vecfn = vecfn

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)

    def act(self, obs):
        torch = self.torch
        from dataset_v2 import normalize_obs
        o = np.ascontiguousarray(obs['observation'])
        vec = self._vecfn(o).astype(np.float32)
        with torch.no_grad():
            ot = normalize_obs(torch.from_numpy(o).unsqueeze(0))
            vt = torch.from_numpy(vec).unsqueeze(0)
            mt = torch.from_numpy(np.ascontiguousarray(obs['action_mask'])).float().unsqueeze(0)
            logits, _ = self.model({'observation': ot, 'vec': vt, 'action_mask': mt})
            logits = logits[0]
            if self.temperature <= 0:
                return int(logits.argmax())
            probs = torch.softmax(logits / self.temperature, dim=0).numpy()
            probs = probs * np.asarray(obs['action_mask'])
            probs /= probs.sum()
            return int(self.rng.choices(range(235), weights=probs)[0])

    def clone_shared(self):
        """共享底层 torch model、独立采样 rng 的副本(多座位独立决策种子用)。"""
        c = NNPolicyV2.__new__(NNPolicyV2)
        c.torch = self.torch
        c.model = self.model            # 共享
        c.temperature = self.temperature
        c.rng = random.Random(0)        # 独立 rng
        c._vecfn = self._vecfn
        return c


def make_policy(spec):
    """'random' | 'rule' | 'nn:<ckpt>[@t<T>][@claim]' | 'nnv2:<ckpt>[@t<T>]' -> Policy。"""
    if spec == 'rule':
        return RuleBotPolicy()
    if spec.startswith('random'):
        seed = int(spec.split(':')[1]) if ':' in spec else 0
        return RandomPolicy(seed)
    if spec.startswith('nnv2dual:'):
        from model_v2_dual import MahjongNetV2Dual
        from model_v2 import InputDictAdapterV2
        parts = spec[9:].split('@')
        temperature = 0.0
        for opt in parts[1:]:
            if opt.startswith('t'):
                temperature = float(opt[1:])
        net, _ = MahjongNetV2Dual.load(parts[0])
        return NNPolicyV2(InputDictAdapterV2(net), temperature=temperature)
    if spec.startswith('nnv2ext:'):
        from model_v2 import MahjongNetV2, InputDictAdapterV2
        from feature_v2_ext import compute_vec_ext_from_obs
        parts = spec[8:].split('@')
        temperature = 0.0
        for opt in parts[1:]:
            if opt.startswith('t'):
                temperature = float(opt[1:])
        net, _ = MahjongNetV2.load(parts[0])
        return NNPolicyV2(InputDictAdapterV2(net), temperature=temperature,
                          vecfn=compute_vec_ext_from_obs)
    if spec.startswith('nnv2:'):
        from model_v2 import MahjongNetV2, InputDictAdapterV2
        parts = spec[5:].split('@')
        temperature = 0.0
        for opt in parts[1:]:
            if opt.startswith('t'):
                temperature = float(opt[1:])
        net, _ = MahjongNetV2.load(parts[0])
        return NNPolicyV2(InputDictAdapterV2(net), temperature=temperature)
    if spec.startswith('nn:'):
        from model_v1 import MahjongNetV1, InputDictAdapter
        parts = spec[3:].split('@')
        path = parts[0]
        temperature, claim = 0.0, False
        for opt in parts[1:]:
            if opt.startswith('t'):
                temperature = float(opt[1:])
            elif opt == 'claim':
                claim = True
        net, _ = MahjongNetV1.load(path)
        return NNPolicy(InputDictAdapter(net), temperature=temperature,
                        claim_tenpai=claim)
    raise ValueError('unknown policy spec %r' % spec)
