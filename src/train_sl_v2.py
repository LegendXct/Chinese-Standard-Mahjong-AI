"""gen2 阶段一 SL-v2 训练:双输入(obs+vec)行为克隆 + 价值头。

与 M2(train_sl.py)同配方便于对比:masked CE + 合法位 smoothing ε=0.05、
价值 SmoothL1×0.5、AdamW、warmup2%+cosine、clip1.0、bf16、batch1024、20ep。
新增:vec 分支、可配 norm(bn/gn)、--exclude-asym(M3 增广去噪)、Play 桶准确率。

用法:python src/train_sl_v2.py --config B --norm bn --epochs 20 \
    --data .../data_v1 --run-dir .../checkpoints/sl_v2_M1_bn [--exclude-asym]
"""
import argparse
import csv
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
from dataset_v2 import MahjongV2Dataset, normalize_obs  # noqa: E402
from model_v2 import MahjongNetV2, CONFIGS_V2  # noqa: E402
import feature_v1 as fv1  # noqa: E402

SEED = 20260610
OA = fv1.FeatureAgentV1.OFFSET_ACT
PLAY_LO, PLAY_HI = OA['Play'], OA['Chi']   # Play 桶 [2,36)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_ce_loss(logits, mask, act, eps=0.05):
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
    play_n = play_correct = 0
    vloss_sum = policy_sum = 0.0
    for obs, vec, mask, act, score in loader:
        obs = normalize_obs(obs.to(device, non_blocking=True))
        vec = vec.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True).float()
        act = act.to(device, non_blocking=True)
        score = score.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, value = model(obs, vec)
            ploss, masked_logits = masked_ce_loss(logits.float(), mask, act)
            vloss = F.smooth_l1_loss(value.float(), score)
        pred = masked_logits.argmax(dim=1)
        hit = (pred == act)
        correct += hit.sum().item()
        n += len(act)
        play_mask = (act >= PLAY_LO) & (act < PLAY_HI)
        play_n += play_mask.sum().item()
        play_correct += (hit & play_mask).sum().item()
        vloss_sum += vloss.item() * len(act)
        policy_sum += ploss.item() * len(act)
    model.train()
    return (correct / n, play_correct / max(play_n, 1),
            policy_sum / n, vloss_sum / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', choices=list(CONFIGS_V2), default='B')
    ap.add_argument('--norm', choices=['gn', 'bn'], default=None)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--data', default='/root/autodl-tmp/mahjong/data_v1')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--exclude-asym', action='store_true', help='绿一色/推不倒禁用增广(M3)')
    ap.add_argument('--vec-ext', action='store_true', help='用扩展 vec(feature_v2_ext,117 维)')
    ap.add_argument('--act', choices=['relu', 'gelu'], default='relu')
    ap.add_argument('--dual', action='store_true', help='用 Arthals 双输入结构 model_v2_dual')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no-compile', action='store_true')
    ap.add_argument('--limit-steps', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    set_seed(SEED)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    cfg = dict(CONFIGS_V2[args.config])
    if args.norm:
        cfg['norm'] = args.norm
    cfg['act'] = args.act
    if args.vec_ext:
        from feature_v2_ext import VEC_DIM_EXT
        cfg['vec_dim'] = VEC_DIM_EXT
    if args.dual:
        from model_v2_dual import MahjongNetV2Dual
        model = MahjongNetV2Dual.from_config(cfg).to(device)
    else:
        model = MahjongNetV2.from_config(cfg).to(device)
    print('config %s %s dual=%s: %.2fM params' % (
        args.config, cfg, args.dual, model.n_params() / 1e6), flush=True)

    train_ds = MahjongV2Dataset(args.data, 'train', augment=True,
                                exclude_asym=args.exclude_asym, return_score=True,
                                ext=args.vec_ext)
    val_ds = MahjongV2Dataset(args.data, 'val', augment=False, return_score=True,
                              ext=args.vec_ext)
    print('train %d / val %d; exclude_asym=%s aug-disabled samples=%d' % (
        len(train_ds), len(val_ds), args.exclude_asym,
        int((~train_ds.can_aug[train_ds.indices]).sum())), flush=True)
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
    if not args.no_compile:
        try:
            compiled = torch.compile(model)
        except Exception as e:
            print('compile failed: %r' % e, flush=True)

    start_epoch, gstep, best_acc = 0, 0, -1.0
    last_path = os.path.join(args.run_dir, 'last.pt')
    if args.resume and os.path.exists(last_path):
        payload = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict({k: v.float() for k, v in payload['state_dict'].items()})
        opt.load_state_dict(payload['optimizer'])
        start_epoch = payload['epoch'] + 1
        gstep = payload['gstep']
        best_acc = payload.get('best_acc', -1.0)
        print('resumed epoch %d' % start_epoch, flush=True)

    ef = open(os.path.join(args.run_dir, 'epoch_log.csv'), 'a', newline='')
    ew = csv.writer(ef)
    if ef.tell() == 0:
        ew.writerow(['epoch', 'val_acc', 'val_play_acc', 'val_policy', 'val_value',
                     'train_loss', 'elapsed_s'])

    model.train()
    t_run = time.time()
    for epoch in range(start_epoch, args.epochs):
        t_ep = time.time()
        t_win = time.time()
        loss_v = 0.0
        for i, (obs, vec, mask, act, score) in enumerate(train_loader):
            if args.limit_steps and i >= args.limit_steps:
                break
            lr = lr_at(gstep)
            for g in opt.param_groups:
                g['lr'] = lr
            obs = normalize_obs(obs.to(device, non_blocking=True))
            vec = vec.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True).float()
            act = act.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, value = compiled(obs, vec)
                ploss, _ = masked_ce_loss(logits.float(), mask, act)
                vloss = F.smooth_l1_loss(value.float(), score)
                loss = ploss + 0.5 * vloss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gstep += 1
            loss_v = loss.item()
            if gstep % 1000 == 0:
                it_s = 1000 / (time.time() - t_win)
                t_win = time.time()
                print('ep %d step %d/%d loss %.4f lr %.2e %.1f it/s gpu %d%%' % (
                    epoch, i + 1, steps_per_epoch, loss_v, lr, it_s, gpu_util()),
                    flush=True)

        val_acc, val_play, val_p, val_v = evaluate(compiled, val_loader, device)
        ew.writerow([epoch, '%.5f' % val_acc, '%.5f' % val_play, '%.4f' % val_p,
                     '%.4f' % val_v, '%.4f' % loss_v, '%.1f' % (time.time() - t_ep)])
        ef.flush()
        print('== epoch %d: val_acc %.5f play_acc %.5f val_value %.4f (%.1f min)' % (
            epoch, val_acc, val_play, val_v, (time.time() - t_ep) / 60), flush=True)

        extra = {'optimizer': opt.state_dict(), 'epoch': epoch, 'gstep': gstep,
                 'best_acc': max(best_acc, val_acc), 'val_acc': val_acc,
                 'val_play_acc': val_play, 'train_args': vars(args)}
        model.save(last_path, extra=extra)
        if val_acc > best_acc:
            best_acc = val_acc
            model.save(os.path.join(args.run_dir, 'best.pt'),
                       extra={'epoch': epoch, 'val_acc': val_acc,
                              'val_play_acc': val_play, 'train_args': vars(args)})

    print('total %.1f min, best val acc %.5f' % ((time.time() - t_run) / 60, best_acc),
          flush=True)
    ef.close()
    print('TRAIN_SL_V2_DONE %s best=%.5f' % (args.run_dir, best_acc))


if __name__ == '__main__':
    main()
