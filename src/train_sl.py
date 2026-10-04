"""M2 监督学习训练:行为克隆 + 辅助价值头。

- 策略损失:masked CE(logits + (1-mask)·(-1e9)),label smoothing ε=0.05
  只分摊到合法动作:target = (1-ε)·onehot + ε·mask/mask.sum()
- 价值损失:目标 = 本局自家最终得分/30,SmoothL1,系数 0.5
- AdamW(norm/bias 免 wd)、warmup 2% + cosine 至 1e-5、clip 1.0、bf16 autocast
- 每 epoch 全量 val + checkpoint(best by val top-1 + last),--resume 续训

用法(服务器,tmux):
  python src/train_sl.py --config A --norm gn --epochs 20 \
      --data /root/autodl-tmp/mahjong/data_v1 --run-dir /root/autodl-tmp/mahjong/checkpoints/sl_A_gn
"""
import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_v1 import MahjongV1Dataset, normalize_obs  # noqa: E402
from model_v1 import MahjongNetV1, CONFIGS  # noqa: E402

SEED = 20260610


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_ce_loss(logits, mask, act, eps=0.05):
    """masked CE + 合法位 label smoothing。mask: float (B,235)。"""
    masked_logits = logits + (1.0 - mask) * (-1e9)
    logp = F.log_softmax(masked_logits, dim=1)
    n_legal = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    target = mask * (eps / n_legal)
    target.scatter_add_(1, act.unsqueeze(1),
                        torch.full_like(act, 1, dtype=target.dtype).unsqueeze(1) * (1.0 - eps))
    return -(target * logp).sum(dim=1).mean(), masked_logits


def gpu_util():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = correct = 0
    vloss_sum = policy_sum = 0.0
    for obs, mask, act, score in loader:
        obs = normalize_obs(obs.to(device, non_blocking=True))
        mask = mask.to(device, non_blocking=True).float()
        act = act.to(device, non_blocking=True)
        score = score.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, value = model(obs)
            ploss, masked_logits = masked_ce_loss(logits.float(), mask, act)
            vloss = F.smooth_l1_loss(value.float(), score)
        pred = masked_logits.argmax(dim=1)
        correct += (pred == act).sum().item()
        n += len(act)
        vloss_sum += vloss.item() * len(act)
        policy_sum += ploss.item() * len(act)
    model.train()
    return correct / n, policy_sum / n, vloss_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', choices=list(CONFIGS), default='A')
    ap.add_argument('--norm', choices=['gn', 'bn'], default=None)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--data', default='/root/autodl-tmp/mahjong/data_v1')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no-compile', action='store_true')
    ap.add_argument('--limit-steps', type=int, default=0, help='冒烟用,每 epoch 截断步数')
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    set_seed(SEED)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    cfg = dict(CONFIGS[args.config])
    if args.norm:
        cfg['norm'] = args.norm
    model = MahjongNetV1.from_config(cfg).to(device)
    n_params = model.n_params()
    print('config %s %s: %.2fM params (fp32 %.1f MB, fp16 %.1f MB)' % (
        args.config, cfg, n_params / 1e6, n_params * 4 / 2**20, n_params * 2 / 2**20),
        flush=True)

    train_ds = MahjongV1Dataset(args.data, 'train', augment=True, return_score=True)
    val_ds = MahjongV1Dataset(args.data, 'val', augment=False, return_score=True)
    sc = train_ds.score[train_ds.indices]
    print('value target (train): min %.2f max %.2f mean %.4f std %.4f' % (
        sc.min(), sc.max(), sc.mean(), sc.std()), flush=True)
    gen = torch.Generator()
    gen.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=True, prefetch_factor=4, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.wd},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=args.lr)

    steps_per_epoch = (len(train_ds) // args.batch) if not args.limit_steps \
        else min(args.limit_steps, len(train_ds) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.02))
    min_lr = 1e-5

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * t))

    compiled = model
    compile_ok = False
    if not args.no_compile:
        try:
            compiled = torch.compile(model)
            compile_ok = True
        except Exception as e:
            print('torch.compile failed, fallback eager: %r' % e, flush=True)

    start_epoch, gstep, best_acc = 0, 0, -1.0
    last_path = os.path.join(args.run_dir, 'last.pt')
    if args.resume and os.path.exists(last_path):
        payload = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict({k: v.float() for k, v in payload['state_dict'].items()})
        opt.load_state_dict(payload['optimizer'])
        start_epoch = payload['epoch'] + 1
        gstep = payload['gstep']
        best_acc = payload.get('best_acc', -1.0)
        print('resumed from epoch %d (gstep %d, best %.4f)' % (start_epoch, gstep, best_acc),
              flush=True)

    csv_path = os.path.join(args.run_dir, 'train_log.csv')
    csv_f = open(csv_path, 'a', newline='')
    csv_w = csv.writer(csv_f)
    if csv_f.tell() == 0:
        csv_w.writerow(['gstep', 'epoch', 'loss', 'policy', 'value', 'lr', 'it_s', 'gpu_util'])
    epoch_csv = os.path.join(args.run_dir, 'epoch_log.csv')
    ef = open(epoch_csv, 'a', newline='')
    ew = csv.writer(ef)
    if ef.tell() == 0:
        ew.writerow(['epoch', 'val_acc', 'val_policy', 'val_value', 'train_loss', 'elapsed_s'])

    model.train()
    t_run = time.time()
    for epoch in range(start_epoch, args.epochs):
        t_ep = time.time()
        t_win = time.time()
        loss_sum = psum = vsum = nb = 0.0
        for i, (obs, mask, act, score) in enumerate(train_loader):
            if args.limit_steps and i >= args.limit_steps:
                break
            lr = lr_at(gstep)
            for g in opt.param_groups:
                g['lr'] = lr
            obs = normalize_obs(obs.to(device, non_blocking=True))
            mask = mask.to(device, non_blocking=True).float()
            act = act.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, value = compiled(obs)
                ploss, _ = masked_ce_loss(logits.float(), mask, act)
                vloss = F.smooth_l1_loss(value.float(), score)
                loss = ploss + 0.5 * vloss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gstep += 1
            loss_sum += loss.item()
            psum += ploss.item()
            vsum += vloss.item()
            nb += 1
            if gstep % 100 == 0:
                it_s = 100 / (time.time() - t_win)
                t_win = time.time()
                util = gpu_util() if gstep % 500 == 0 else ''
                csv_w.writerow(['%d' % gstep, epoch, '%.4f' % (loss_sum / nb),
                                '%.4f' % (psum / nb), '%.4f' % (vsum / nb),
                                '%.2e' % lr, '%.2f' % it_s, util])
                csv_f.flush()
                if gstep % 1000 == 0:
                    print('ep %d step %d/%d loss %.4f lr %.2e %.1f it/s gpu %s%%' % (
                        epoch, i + 1, steps_per_epoch, loss_sum / nb, lr, it_s, util),
                        flush=True)
                loss_sum = psum = vsum = nb = 0.0

        val_acc, val_p, val_v = evaluate(compiled, val_loader, device)
        ew.writerow([epoch, '%.5f' % val_acc, '%.4f' % val_p, '%.4f' % val_v,
                     '%.4f' % (loss.item()), '%.1f' % (time.time() - t_ep)])
        ef.flush()
        print('== epoch %d done: val_acc %.4f val_policy %.4f val_value %.4f (%.1f min)' % (
            epoch, val_acc, val_p, val_v, (time.time() - t_ep) / 60), flush=True)

        extra = {'optimizer': opt.state_dict(), 'epoch': epoch, 'gstep': gstep,
                 'best_acc': max(best_acc, val_acc), 'val_acc': val_acc,
                 'train_args': vars(args)}
        model.save(last_path, extra=extra)
        if val_acc > best_acc:
            best_acc = val_acc
            model.save(os.path.join(args.run_dir, 'best.pt'),
                       extra={'epoch': epoch, 'val_acc': val_acc, 'train_args': vars(args)})

    print('total %.1f min, best val acc %.4f, compile=%s' % (
        (time.time() - t_run) / 60, best_acc, compile_ok), flush=True)
    csv_f.close()
    ef.close()


if __name__ == '__main__':
    main()
