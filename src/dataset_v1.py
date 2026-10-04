"""特征 v1 数据集:memmap 加载 + 在线花色置换增广。

置换语义:SUIT_PERMS[k] = σ,旧花色 s(0=W,1=T,2=B)的牌改名为花色 σ[s]。
- obs:所有通道的第 0~2 行按 σ 整体搬移(第 3 行字牌不动);
- act:new_act = ACT_PERMS[k][old_act];
- mask:new_mask = old_mask[MASK_GATHER[k]](MASK_GATHER = ACT_PERMS 的逆置换)。

标量通道(×255 存储)用 CHANNEL_SCALE 在训练侧还原:obs.float() * CHANNEL_SCALE。
"""
import itertools
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import feature_v1 as fv1

# ---------------- 置换表 ----------------
SUIT_PERMS = list(itertools.permutations(range(3)))  # 6 种,[0]=恒等


def _build_tile_perm(sigma):
    """34 维牌索引置换:P34[旧牌] = 新牌。"""
    p = np.arange(34)
    for s in range(3):
        for n in range(9):
            p[s * 9 + n] = sigma[s] * 9 + n
    return p


def _build_act_perm(sigma):
    """235 维动作置换:P235[旧动作] = 新动作。"""
    OA = fv1.FeatureAgentV1.OFFSET_ACT
    p34 = _build_tile_perm(sigma)
    p = np.arange(235)
    for base in (OA['Play'], OA['Peng'], OA['Gang'], OA['AnGang'], OA['BuGang']):
        for i in range(34):
            p[base + i] = base + p34[i]
    # Chi 区:36 + 花色*21 + (中心牌-2)*3 + 吃位
    for s in range(3):
        for m in range(7):
            for pos in range(3):
                p[OA['Chi'] + s * 21 + m * 3 + pos] = OA['Chi'] + sigma[s] * 21 + m * 3 + pos
    return p


def _build_row_gather(sigma):
    """4 行 gather 索引:new_obs[:, r, :] = obs[:, G[r], :]。"""
    ext = np.array([sigma[0], sigma[1], sigma[2], 3])
    return np.argsort(ext)


TILE_PERMS = np.stack([_build_tile_perm(s) for s in SUIT_PERMS])
ACT_PERMS = np.stack([_build_act_perm(s) for s in SUIT_PERMS])
MASK_GATHER = np.stack([np.argsort(p) for p in ACT_PERMS])
ROW_GATHER = np.stack([_build_row_gather(s) for s in SUIT_PERMS])

# 训练侧还原系数:×255 的标量通道乘 1/255,其余为 1
CHANNEL_SCALE = np.ones(fv1.NUM_CHANNELS, np.float32)
CHANNEL_SCALE[list(fv1.SCALED_CHANNELS)] = 1.0 / 255
CHANNEL_SCALE_T = torch.from_numpy(CHANNEL_SCALE).view(-1, 1, 1)


def augment_sample(obs, mask, act, k):
    """对单样本应用第 k 种花色置换(k=0 恒等)。obs (C,4,9), mask (235,), act int。"""
    if k == 0:
        return obs, mask, act
    return obs[:, ROW_GATHER[k], :], mask[MASK_GATHER[k]], int(ACT_PERMS[k][act])


class MahjongV1Dataset(Dataset):
    """split: 'train' | 'val' | 'all'。augment=True 时每样本随机取 6 种置换之一。

    return_score=True 时附带返回价值目标:本局自家最终原始得分 / score_scale
    (来自 meta.npz 的 score;花色置换不改变得分)。
    """

    def __init__(self, root, split='train', augment=False,
                 return_score=False, score_scale=30.0):
        with open(os.path.join(root, 'info.json')) as f:
            self.info = json.load(f)
        assert self.info['feature_version'] == fv1.FEATURE_VERSION
        self.obs = np.load(os.path.join(root, 'obs.dat'), mmap_mode='r')
        self.mask = np.load(os.path.join(root, 'mask.dat'), mmap_mode='r')
        self.act = np.load(os.path.join(root, 'act.dat'), mmap_mode='r')
        if split == 'all':
            self.indices = np.arange(len(self.act))
        else:
            sp = np.load(os.path.join(root, 'split.npz'))
            self.indices = sp['%s_idx' % split]
        self.augment = augment
        self.return_score = return_score
        if return_score:
            with np.load(os.path.join(root, 'meta.npz')) as meta:
                self.score = (meta['score'].astype(np.float32) / score_scale)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        obs = np.array(self.obs[idx])    # 从 memmap 拷出,避免只读 tensor 警告
        mask = np.array(self.mask[idx])
        act = int(self.act[idx])
        if self.augment:
            k = random.randrange(6)
            obs2, mask2, act2 = augment_sample(obs, mask, act, k)
            # 极罕见:动作为 Hu 且花色绑定番种(绿一色/推不倒)在置换后跌破 8 番,
            # 置换后样本自相矛盾(act 不在 mask 内),此时放弃增广用原样本
            if mask2[act2]:
                obs, mask, act = obs2, mask2, act2
        items = (torch.from_numpy(np.ascontiguousarray(obs)),
                 torch.from_numpy(np.ascontiguousarray(mask)),
                 act)
        if self.return_score:
            return items + (self.score[idx],)
        return items


def normalize_obs(obs_uint8_tensor):
    """uint8 obs (B,C,4,9) -> float32,×255 标量通道还原为 [0,1]。"""
    scale = CHANNEL_SCALE_T.to(obs_uint8_tensor.device)
    return obs_uint8_tensor.float() * scale
