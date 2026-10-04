"""dataloader 吞吐基准:batch=1024,含增广与拷贝到 GPU。

用法(服务器项目根目录):
  python scripts/bench_loader.py --root /root/autodl-tmp/mahjong/data_v1 --workers 16
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from dataset_v1 import MahjongV1Dataset, normalize_obs  # noqa: E402


def bench(loader, n_batches, device, label):
    it = iter(loader)
    # 预热(含 worker 启动)
    for _ in range(8):
        next(it)
    t0 = time.time()
    n = 0
    for _ in range(n_batches):
        try:
            obs, mask, act = next(it)
        except StopIteration:
            break
        obs = normalize_obs(obs.to(device, non_blocking=True))
        mask = mask.to(device, non_blocking=True)
        act = act.to(device, non_blocking=True)
        n += len(act)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0
    print('%s: %d samples in %.2fs -> %.0f samples/s' % (label, n, dt, n / dt))
    return n / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--batches', type=int, default=300)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for augment in (False, True):
        ds = MahjongV1Dataset(args.root, split='train', augment=augment)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0, drop_last=True,
                            prefetch_factor=4 if args.workers > 0 else None)
        bench(loader, args.batches, device, 'augment=%s workers=%d' % (augment, args.workers))
        del loader


if __name__ == '__main__':
    main()
