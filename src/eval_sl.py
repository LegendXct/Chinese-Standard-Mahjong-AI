"""M2 评估:分桶准确率 / 价值头指标 / 类型混淆矩阵。

用法:
  python src/eval_sl.py --ckpt /root/autodl-tmp/mahjong/checkpoints/sl_A_gn/best.pt \
      --data /root/autodl-tmp/mahjong/data_v1 --out results/eval_sl_A.md
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_v1 import MahjongV1Dataset, normalize_obs  # noqa: E402
from model_v1 import MahjongNetV1  # noqa: E402

TYPES = [('Pass', 0, 1), ('Hu', 1, 2), ('Play', 2, 36), ('Chi', 36, 99),
         ('Peng', 99, 133), ('Gang', 133, 167), ('AnGang', 167, 201), ('BuGang', 201, 235)]


def act_type(a):
    for i, (name, lo, hi) in enumerate(TYPES):
        if lo <= a < hi:
            return i
    raise ValueError(a)


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    type_n = np.zeros(8, np.int64)
    type_correct = np.zeros(8, np.int64)
    confusion = np.zeros((8, 8), np.int64)       # [label_type, pred_type]
    play_top3_n = play_top3_c = 0
    grp_n = np.zeros(2, np.int64)                # 0=出牌态, 1=响应态
    grp_c = np.zeros(2, np.int64)
    v_pred_all, v_tgt_all = [], []
    sl1_sum = mae_sum = n_all = 0.0

    type_idx = np.array([act_type(a) for a in range(235)])
    type_idx_t = torch.from_numpy(type_idx).to(device)

    for obs, mask, act, score in loader:
        obs = normalize_obs(obs.to(device, non_blocking=True))
        mask = mask.to(device, non_blocking=True).float()
        act = act.to(device, non_blocking=True)
        score = score.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, value = model(obs)
        logits = logits.float() + (1.0 - mask) * (-1e9)
        value = value.float()
        pred = logits.argmax(dim=1)

        lt = type_idx_t[act]
        pt = type_idx_t[pred]
        for L in range(8):
            sel = lt == L
            if sel.any():
                type_n[L] += sel.sum().item()
                type_correct[L] += (pred[sel] == act[sel]).sum().item()
                idx, cnt = torch.unique(pt[sel], return_counts=True)
                for i_, c_ in zip(idx.tolist(), cnt.tolist()):
                    confusion[L, i_] += c_
        # Play 桶 top-3
        sel = lt == 2
        if sel.any():
            top3 = logits[sel].topk(3, dim=1).indices
            play_top3_c += (top3 == act[sel].unsqueeze(1)).any(dim=1).sum().item()
            play_top3_n += sel.sum().item()
        # 出牌态(mask 含 Play 动作)vs 响应态
        is_play_state = mask[:, 2:36].any(dim=1).long()  # 1=出牌态
        for g in (0, 1):
            sel = is_play_state == (1 - g)  # g=0 出牌态
            if sel.any():
                grp_n[g] += sel.sum().item()
                grp_c[g] += (pred[sel] == act[sel]).sum().item()
        # 价值头
        sl1_sum += F.smooth_l1_loss(value, score, reduction='sum').item()
        mae_sum += (value - score).abs().sum().item()
        n_all += len(act)
        v_pred_all.append(value.cpu().numpy())
        v_tgt_all.append(score.cpu().numpy())

    v_pred = np.concatenate(v_pred_all)
    v_tgt = np.concatenate(v_tgt_all)
    pearson = float(np.corrcoef(v_pred, v_tgt)[0, 1])
    return {
        'total_acc': float(type_correct.sum() / type_n.sum()),
        'type_n': type_n, 'type_correct': type_correct, 'confusion': confusion,
        'play_top3': play_top3_c / max(play_top3_n, 1),
        'acc_play_state': grp_c[0] / max(grp_n[0], 1),
        'acc_resp_state': grp_c[1] / max(grp_n[1], 1),
        'value_smooth_l1': sl1_sum / n_all, 'value_mae': mae_sum / n_all,
        'value_pearson': pearson,
    }


def to_markdown(r, ckpt, payload):
    L = []
    L.append('# SL 评估:%s' % ckpt)
    L.append('')
    L.append('- checkpoint epoch: %s, val_acc(训练时): %s' % (
        payload.get('epoch'), payload.get('val_acc')))
    L.append('- **val 总 top-1 准确率: %.4f**' % r['total_acc'])
    L.append('- 出牌态: %.4f | 响应态: %.4f | Play 桶 top-3: %.4f' % (
        r['acc_play_state'], r['acc_resp_state'], r['play_top3']))
    L.append('- 价值头: SmoothL1 %.4f | MAE %.4f | Pearson %.4f' % (
        r['value_smooth_l1'], r['value_mae'], r['value_pearson']))
    L.append('')
    L.append('| 动作类型 | 样本数 | top-1 acc |')
    L.append('|---|---|---|')
    for i, (name, _, _) in enumerate(TYPES):
        n = r['type_n'][i]
        L.append('| %s | %d | %s |' % (name, n,
                 '%.4f' % (r['type_correct'][i] / n) if n else '-'))
    L.append('')
    L.append('## 类型混淆矩阵(行=标签,列=预测)')
    L.append('')
    L.append('| 标签\\预测 | ' + ' | '.join(t[0] for t in TYPES) + ' |')
    L.append('|---|' + '---|' * 8)
    for i, (name, _, _) in enumerate(TYPES):
        L.append('| %s | ' % name + ' | '.join(str(x) for x in r['confusion'][i]) + ' |')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='/root/autodl-tmp/mahjong/data_v1')
    ap.add_argument('--out', default=None)
    ap.add_argument('--batch', type=int, default=2048)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, payload = MahjongNetV1.load(args.ckpt)
    model = model.to(device)
    ds = MahjongV1Dataset(args.data, 'val', augment=False, return_score=True)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=8, pin_memory=True)
    r = run_eval(model, loader, device)
    md = to_markdown(r, args.ckpt, payload)
    print(md)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(md + '\n')


if __name__ == '__main__':
    main()
