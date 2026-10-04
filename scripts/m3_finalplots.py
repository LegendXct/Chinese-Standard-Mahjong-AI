# -*- coding: utf-8 -*-
"""M3.2 最终报告曲线:train_metrics + eval_full -> 2 张 PNG(x 轴=update,跨 resume 单调)。"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

EVENTS = [
    (248, 'pilot->main'),
    (541, 'lr restore 4e-5'),
    (735, 'infer crash+resume'),
    (1073, 'lr->6e-5'),
    (1746, 'eval_best'),
    (2425, 'early stop'),
]


def read_csv(path, header=None):
    with open(path) as f:
        if header:
            return [dict(zip(header, row)) for row in csv.reader(f)]
        return list(csv.DictReader(f))


def vlines(ax, upds_only=False):
    for u, label in EVENTS:
        ax.axvline(u, color='red' if 'crash' in label else 'gray',
                   lw=0.8, ls='--', alpha=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='/root/autodl-tmp/mahjong/checkpoints/rl_v1')
    ap.add_argument('--outdir', default='/root/autodl-tmp/mahjong/results')
    args = ap.parse_args()
    tm = read_csv(os.path.join(args.run, 'train_metrics.csv'))
    ef = read_csv(os.path.join(args.run, 'eval_full.csv'),
                  header=['time_min', 'update', 'games', 'sl_d', 'sl_se',
                          'hu', 'ru_d', 'ru_se', 'x9'])
    u = [int(r['update']) for r in tm]

    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))

    def p(a, key, title, extra_key=None, labels=None):
        a.plot(u, [float(r[key]) for r in tm], lw=0.9)
        if extra_key:
            a.plot(u, [float(r[extra_key]) for r in tm], lw=0.9, alpha=0.65)
            a.legend(labels, fontsize=8)
        a.set_title(title)
        a.set_xlabel('update')
        a.grid(alpha=0.3)
        vlines(a)

    p(ax[0][0], 'entropy', 'entropy')
    p(ax[0][1], 'kl_sl', 'KL vs anchor (=sl0, 0 reanchors)')
    p(ax[0][2], 'clip_frac', 'clip_frac')
    p(ax[1][0], 'hu_rate', 'selfplay hu / huang', 'huang_rate', ['hu', 'huang'])
    p(ax[1][1], 'avg_fan', 'selfplay avg fan')
    p(ax[1][2], 'games_per_s', 'throughput games/s')
    for a, (uu, label) in zip([ax[0][0]] * 6, []):
        pass
    # 事件图例写在标题下
    fig.suptitle('M3.2 PPO training curves  |  events: '
                 + ', '.join('%d=%s' % e for e in EVENTS), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = os.path.join(args.outdir, 'm3_train_curves.png')
    fig.savefig(out1, dpi=130)

    fig2, a2 = plt.subplots(1, 2, figsize=(13, 4.6))
    ue = [int(r['update']) for r in ef]
    a2[0].errorbar(ue, [float(r['sl_d']) for r in ef],
                   yerr=[float(r['sl_se']) for r in ef], marker='o', capsize=3, lw=1)
    a2[0].axhline(0, color='gray', lw=0.8)
    a2[0].set_title('full eval vs SL champion (pair diff ±SE, 2000 games)')
    a2[1].errorbar(ue, [float(r['ru_d']) for r in ef],
                   yerr=[float(r['ru_se']) for r in ef], marker='s', capsize=3,
                   lw=1, color='darkgreen')
    a2[1].axhline(6.811, color='gray', lw=0.8, ls=':')
    a2[1].set_title('full eval vs RuleBot (±SE; dotted = SL anchor +6.81)')
    for a in a2:
        a.set_xlabel('update')
        a.grid(alpha=0.3)
        a.axvline(1746, color='gray', lw=0.8, ls='--')
        a.axvline(2425, color='red', lw=0.8, ls='--')
    fig2.tight_layout()
    out2 = os.path.join(args.outdir, 'm3_eval_curves.png')
    fig2.savefig(out2, dpi=130)
    print('saved', out1, out2)


if __name__ == '__main__':
    main()
