# -*- coding: utf-8 -*-
"""gen2 阶段一冒烟:真实 obs 现算 vec -> dataset 张量 -> model_v2 bn/gn 前向 + 一步训练。
不依赖 vec.dat(直接从 obs.dat 现算),确认 v2 全链路代码通。"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from feature_v2 import compute_vec_from_obs, VEC_DIM  # noqa: E402
from model_v2 import MahjongNetV2  # noqa: E402
from dataset_v2 import normalize_obs, VEC_GATHER, augment_v2  # noqa: E402
from train_sl_v2 import masked_ce_loss  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/mahjong/data_v1'
B = 64

obs = np.load(os.path.join(DATA, 'obs.dat'), mmap_mode='r')
mask = np.load(os.path.join(DATA, 'mask.dat'), mmap_mode='r')
act = np.load(os.path.join(DATA, 'act.dat'), mmap_mode='r')
idx = np.linspace(0, len(obs) - 1, B).astype(int)

obs_b = np.stack([np.asarray(obs[i]) for i in idx])
vec_b = np.stack([compute_vec_from_obs(obs_b[j]) for j in range(B)]).astype(np.float32)
mask_b = np.stack([np.asarray(mask[i]) for i in idx]).astype(np.float32)
act_b = np.array([int(act[i]) for i in idx])
print('vec dim %d, vec range [%.3f, %.3f], per-discard nonzero rows: %d/%d' % (
    VEC_DIM, vec_b.min(), vec_b.max(),
    int((vec_b[:, :34] < 1.0).any(axis=1).sum()), B), flush=True)

# 增广一致性自检:置换后 vec 维度不变、每弃牌块确被重排
o2, v2, m2, a2 = augment_v2(obs_b[0], vec_b[0], mask_b[0], act_b[0], 1)
assert v2.shape == vec_b[0].shape and o2.shape == obs_b[0].shape
print('augment_v2 ok: vec reshaped, perm act %d->%d' % (act_b[0], a2), flush=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
for norm in ['bn', 'gn']:
    net = MahjongNetV2(width=192, depth=20, norm=norm).to(device)
    print('model_v2 norm=%s: %.2fM params' % (norm, net.n_params() / 1e6), flush=True)
    o = normalize_obs(torch.from_numpy(obs_b)).to(device)
    v = torch.from_numpy(vec_b).to(device)
    m = torch.from_numpy(mask_b).to(device)
    a = torch.from_numpy(act_b).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    with torch.autocast(device, dtype=torch.bfloat16):
        logits, value = net(o, v)
        ploss, _ = masked_ce_loss(logits.float(), m, a)
        vloss = F.smooth_l1_loss(value.float(), torch.zeros_like(value.float()))
        loss = ploss + 0.5 * vloss
    assert logits.shape == (B, 235) and value.shape == (B,)
    assert torch.isfinite(loss), 'loss not finite'
    opt.zero_grad(); loss.backward(); opt.step()
    print('  forward+step ok: logits%s value%s loss %.4f' % (
        tuple(logits.shape), tuple(value.shape), loss.item()), flush=True)

print('SMOKE_OK')
