# -*- coding: utf-8 -*-
"""给 v1 模型(M0/M1)补算 val 总准确率 + Play 桶准确率,口径与 train_sl.evaluate 严格一致
(masked argmax,Play 桶 = act∈[OFFSET_ACT['Play'], OFFSET_ACT['Chi']))。
用法:python scripts/eval_play_bucket.py <ckpt> [data_dir]"""
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from dataset_v1 import MahjongV1Dataset, normalize_obs  # noqa: E402
from model_v1 import MahjongNetV1  # noqa: E402
import feature_v1 as fv1  # noqa: E402

OA = fv1.FeatureAgentV1.OFFSET_ACT
PLAY_LO, PLAY_HI = OA['Play'], OA['Chi']


def main():
    ckpt = sys.argv[1]
    data = sys.argv[2] if len(sys.argv) > 2 else '/root/autodl-tmp/mahjong/data_v1'
    device = 'cuda'
    net, _ = MahjongNetV1.load(ckpt)
    net = net.to(device).eval()
    ds = MahjongV1Dataset(data, 'val', augment=False, return_score=True)
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=8)
    n = correct = pn = pc = 0
    with torch.no_grad():
        for obs, mask, act, score in loader:
            obs = normalize_obs(obs.to(device, non_blocking=True))
            mask = mask.to(device, non_blocking=True).float()
            act = act.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = net(obs)
            ml = logits.float() + (1.0 - mask) * (-1e9)
            hit = (ml.argmax(dim=1) == act)
            correct += hit.sum().item()
            n += len(act)
            pm = (act >= PLAY_LO) & (act < PLAY_HI)
            pn += pm.sum().item()
            pc += (hit & pm).sum().item()
    name = ckpt.split('/')[-2]
    print('%s total_acc=%.5f play_acc=%.5f (n=%d play_n=%d)' % (
        name, correct / n, pc / max(pn, 1), n, pn), flush=True)


if __name__ == '__main__':
    main()
