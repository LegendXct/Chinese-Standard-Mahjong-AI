"""训练曲线绘图:train_log.csv + epoch_log.csv -> PNG。

用法:python scripts/plot_sl.py --runs checkpoints/sl_A_gn checkpoints/sl_B_gn --out results/sl_curves.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in args.runs:
        name = os.path.basename(run.rstrip('/'))
        tl = read_csv(os.path.join(run, 'train_log.csv'))
        el = read_csv(os.path.join(run, 'epoch_log.csv'))
        axes[0].plot([int(r['gstep']) for r in tl], [float(r['loss']) for r in tl],
                     label=name, alpha=0.8, linewidth=0.8)
        axes[1].plot([int(r['epoch']) + 1 for r in el], [float(r['val_acc']) for r in el],
                     marker='o', label=name)
    axes[0].set_xlabel('step')
    axes[0].set_ylabel('train loss')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('val top-1 acc')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print('saved', args.out)


if __name__ == '__main__':
    main()
