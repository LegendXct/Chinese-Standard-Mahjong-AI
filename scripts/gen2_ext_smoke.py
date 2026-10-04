# -*- coding: utf-8 -*-
"""阶段一-C 冒烟:真实 obs 现算扩展 vec(117)-> 增广 gather 自检 -> model_v2(vec_dim=117)
前向 + 一步训练。不依赖 vec_ext.dat。"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from feature_v2_ext import compute_vec_ext_from_obs, VEC_DIM_EXT, vec_gather_ext, BASE_DIM  # noqa: E402
from model_v2 import MahjongNetV2  # noqa: E402
from dataset_v2 import normalize_obs, augment_v2  # noqa: E402
from train_sl_v2 import masked_ce_loss  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/mahjong/data_v1'
B = 64

obs = np.load(os.path.join(DATA, 'obs.dat'), mmap_mode='r')
mask = np.load(os.path.join(DATA, 'mask.dat'), mmap_mode='r')
act = np.load(os.path.join(DATA, 'act.dat'), mmap_mode='r')
idx = np.linspace(0, len(obs) - 1, B).astype(int)

obs_b = np.stack([np.asarray(obs[i]) for i in idx])
vec_b = np.stack([compute_vec_ext_from_obs(obs_b[j]) for j in range(B)]).astype(np.float32)
mask_b = np.stack([np.asarray(mask[i]) for i in idx]).astype(np.float32)
act_b = np.array([int(act[i]) for i in idx])

assert vec_b.shape == (B, VEC_DIM_EXT), vec_b.shape
assert np.isfinite(vec_b).all() and vec_b.min() >= 0.0 and vec_b.max() <= 1.0001, \
    (vec_b.min(), vec_b.max())
ext_nonzero = int((vec_b[:, BASE_DIM:] != 0).any(axis=1).sum())
print('VEC_DIM_EXT=%d range[%.3f,%.3f] ext-nonzero rows %d/%d' % (
    VEC_DIM_EXT, vec_b.min(), vec_b.max(), ext_nonzero, B), flush=True)

# 增广 gather 自检:ext 段必须恒等(花色无关),前 107 段被重排
g = vec_gather_ext()
assert g.shape == (6, VEC_DIM_EXT)
assert (g[:, BASE_DIM:] == np.arange(BASE_DIM, VEC_DIM_EXT)).all(), 'EXT 段未恒等'
o2, v2, m2, a2 = augment_v2(obs_b[0], vec_b[0], mask_b[0], act_b[0], 1, g)
assert v2.shape == (VEC_DIM_EXT,) and (v2[BASE_DIM:] == vec_b[0][BASE_DIM:]).all()
print('aug ok: ext 段置换不变, perm act %d->%d' % (act_b[0], a2), flush=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = MahjongNetV2(width=192, depth=20, norm='gn', vec_dim=VEC_DIM_EXT).to(device)
print('model_v2 vec_dim=%d: %.2fM params' % (VEC_DIM_EXT, net.n_params() / 1e6), flush=True)
o = normalize_obs(torch.from_numpy(obs_b)).to(device)
v = torch.from_numpy(vec_b).to(device)
m = torch.from_numpy(mask_b).to(device)
a = torch.from_numpy(act_b).to(device)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
with torch.autocast(device, dtype=torch.bfloat16):
    logits, value = net(o, v)
    ploss, _ = masked_ce_loss(logits.float(), m, a)
    loss = ploss + 0.5 * F.smooth_l1_loss(value.float(), torch.zeros_like(value.float()))
assert logits.shape == (B, 235) and torch.isfinite(loss)
opt.zero_grad(); loss.backward(); opt.step()
print('forward+step ok: loss %.4f' % loss.item(), flush=True)
print('EXT_SMOKE_OK')
