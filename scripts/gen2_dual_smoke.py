# -*- coding: utf-8 -*-
"""阶段一-D 冒烟:真实 obs+vec → M5(v2+gelu)/M6(dual+relu)/M7(dual+gelu) 前向+一步训练。
确认双输入结构、GELU、价值头、Arthals 下采样 MLP 维度都通,参数量打印。"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from feature_v2 import compute_vec_from_obs  # noqa: E402
from model_v2 import MahjongNetV2  # noqa: E402
from model_v2_dual import MahjongNetV2Dual  # noqa: E402
from dataset_v2 import normalize_obs  # noqa: E402
from train_sl_v2 import masked_ce_loss  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/mahjong/data_v1'
B = 32

obs = np.load(os.path.join(DATA, 'obs.dat'), mmap_mode='r')
mask = np.load(os.path.join(DATA, 'mask.dat'), mmap_mode='r')
act = np.load(os.path.join(DATA, 'act.dat'), mmap_mode='r')
idx = np.linspace(0, len(obs) - 1, B).astype(int)
ob = np.stack([np.asarray(obs[i]) for i in idx])
ve = np.stack([compute_vec_from_obs(ob[j]) for j in range(B)]).astype(np.float32)
mk = np.stack([np.asarray(mask[i]) for i in idx]).astype(np.float32)
ac = np.array([int(act[i]) for i in idx])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
o = normalize_obs(torch.from_numpy(ob)).to(device)
v = torch.from_numpy(ve).to(device)
m = torch.from_numpy(mk).to(device)
a = torch.from_numpy(ac).to(device)

configs = [
    ('M5 v2+gelu', MahjongNetV2(width=192, depth=20, norm='gn', act='gelu')),
    ('M6 dual+relu', MahjongNetV2Dual(width=192, depth=20, norm='gn', act='relu')),
    ('M7 dual+gelu', MahjongNetV2Dual(width=192, depth=20, norm='gn', act='gelu')),
]
for name, net in configs:
    net = net.to(device)
    extra = ' MLP dims %s' % (net.dims,) if hasattr(net, 'dims') else ''
    print('%s: %.2fM params%s' % (name, net.n_params() / 1e6, extra), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    with torch.autocast(device, dtype=torch.bfloat16):
        logits, val = net(o, v)
        ploss, _ = masked_ce_loss(logits.float(), m, a)
        loss = ploss + 0.5 * F.smooth_l1_loss(val.float(), torch.zeros_like(val.float()))
    assert logits.shape == (B, 235) and val.shape == (B,) and torch.isfinite(loss), name
    opt.zero_grad(); loss.backward(); opt.step()
    print('  forward+step ok: loss %.4f' % loss.item(), flush=True)

print('DUAL_SMOKE_OK')
