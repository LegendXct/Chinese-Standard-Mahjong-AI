"""特征 v2 数据集:复用 v1 的 obs/mask/act/split/meta + 新增 vec.dat。

- 空间 obs 与 v1 完全一致(同一份 obs.dat),花色置换沿用 dataset_v1 的置换表;
- vec 的每弃牌块(34×3)按牌种置换(TILE_GATHER),全局标量花色无关不动;
- M3 增广去噪:绿一色/推不倒所在对局的样本禁用增广(can_aug=False,仍参与训练,
  只是恒等——与 Arthals cannot_enhance_matches 同语义),其余样本随机 6 置换。
"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import feature_v1 as fv1
import feature_v2 as fv2
from dataset_v1 import (ROW_GATHER, MASK_GATHER, ACT_PERMS, TILE_PERMS,
                        CHANNEL_SCALE_T)

# vec 花色置换 gather:每弃牌块按 argsort(TILE_PERMS) 重排,全局标量恒等
TILE_GATHER = np.stack([np.argsort(p) for p in TILE_PERMS])
_VEC_GATHER = np.tile(np.arange(fv2.VEC_DIM), (len(TILE_PERMS), 1))
for _k in range(len(TILE_PERMS)):
    for _blk in fv2.PD_BLOCKS:
        _VEC_GATHER[_k, _blk:_blk + 34] = _blk + TILE_GATHER[_k]
VEC_GATHER = _VEC_GATHER


def augment_v2(obs, vec, mask, act, k, vec_gather=VEC_GATHER):
    if k == 0:
        return obs, vec, mask, act
    return (obs[:, ROW_GATHER[k], :], vec[vec_gather[k]],
            mask[MASK_GATHER[k]], int(ACT_PERMS[k][act]))


def load_asym_gids(data_root):
    p = os.path.join(data_root, 'asym_fan_gids.json')
    if os.path.exists(p):
        with open(p) as f:
            return set(json.load(f))
    return set()


class MahjongV2Dataset(Dataset):
    """split: train|val|all。augment=True 随机 6 置换;exclude_asym=True 时
    绿一色/推不倒对局样本禁用增广(恒等)。return_score 附价值目标。"""

    def __init__(self, root, split='train', augment=False, exclude_asym=False,
                 return_score=False, score_scale=30.0, ext=False):
        info_name = 'vec_ext_info.json' if ext else 'vec_info.json'
        want_ver = 'v2.1' if ext else 'v2'
        with open(os.path.join(root, info_name)) as f:
            assert json.load(f)['feature_version'] == want_ver
        self.obs = np.load(os.path.join(root, 'obs.dat'), mmap_mode='r')
        self.mask = np.load(os.path.join(root, 'mask.dat'), mmap_mode='r')
        self.act = np.load(os.path.join(root, 'act.dat'), mmap_mode='r')
        self.vec = np.load(os.path.join(root, 'vec_ext.dat' if ext else 'vec.dat'),
                           mmap_mode='r')
        if ext:
            from feature_v2_ext import vec_gather_ext
            self.vec_gather = vec_gather_ext()
        else:
            self.vec_gather = VEC_GATHER
        if split == 'all':
            self.indices = np.arange(len(self.act))
        else:
            sp = np.load(os.path.join(root, 'split.npz'))
            self.indices = sp['%s_idx' % split]
        self.augment = augment
        self.return_score = return_score
        meta = np.load(os.path.join(root, 'meta.npz'))
        self.gid = meta['gid']
        if return_score:
            self.score = meta['score'].astype(np.float32) / score_scale
        # can_aug[sample] = 该样本是否允许增广
        if exclude_asym:
            asym = load_asym_gids(root)
            self.can_aug = ~np.isin(self.gid, np.array(sorted(asym), dtype=self.gid.dtype)) \
                if asym else np.ones(len(self.gid), dtype=bool)
        else:
            self.can_aug = np.ones(len(self.gid), dtype=bool)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        obs = np.array(self.obs[idx])
        vec = np.array(self.vec[idx]).astype(np.float32)
        mask = np.array(self.mask[idx])
        act = int(self.act[idx])
        if self.augment and self.can_aug[idx]:
            k = random.randrange(6)
            o2, v2, m2, a2 = augment_v2(obs, vec, mask, act, k, self.vec_gather)
            if m2[a2]:           # 置换后动作仍合法才采用(双保险)
                obs, vec, mask, act = o2, v2, m2, a2
        items = (torch.from_numpy(np.ascontiguousarray(obs)),
                 torch.from_numpy(np.ascontiguousarray(vec)),
                 torch.from_numpy(np.ascontiguousarray(mask)),
                 act)
        if self.return_score:
            return items + (self.score[idx],)
        return items


def normalize_obs(obs_uint8_tensor):
    scale = CHANNEL_SCALE_T.to(obs_uint8_tensor.device)
    return obs_uint8_tensor.float() * scale
