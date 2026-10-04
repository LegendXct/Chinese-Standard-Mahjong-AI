"""RL 训练曲线:train_metrics.csv + eval_log.csv -> PNG。

用法:python scripts/plot_rl.py --run /root/autodl-tmp/mahjong/checkpoints/rl_v1 --out results/rl_pilot.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    tm = read(os.path.join(args.run, 'train_metrics.csv'))
    ev = read(os.path.join(args.run, 'eval_log.csv'))

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    t = [float(r['time_min']) for r in tm]

    def plot(a, key, label, scale=1.0):
        a.plot(t, [float(r[key]) * scale for r in tm])
        a.set_title(label)
        a.set_xlabel('min')
        a.grid(alpha=0.3)

    plot(ax[0][0], 'policy', 'policy loss')
    ax[0][0].plot(t, [float(r['value']) for r in tm], label='value')
    ax[0][0].legend(['policy', 'value'])
    plot(ax[0][1], 'entropy', 'entropy')
    plot(ax[0][2], 'kl_sl', 'KL(pi||pi_SL)')
    plot(ax[1][0], 'clip_frac', 'clip_frac')
    plot(ax[1][1], 'hu_rate', 'selfplay hu_rate')
    ax[1][1].plot(t, [float(r['huang_rate']) for r in tm], alpha=0.6)
    ax[1][1].legend(['hu', 'huang'])
    if ev:
        te = [float(r['time_min']) for r in ev]
        pd = [float(r['pair_diff']) for r in ev]
        se = [float(r['pair_se']) for r in ev]
        ax[1][2].errorbar(te, pd, yerr=[2 * s for s in se], marker='o', capsize=3)
        ax[1][2].axhline(0, color='gray', lw=0.8)
        ax[1][2].set_title('eval pair_diff vs SL (±2SE)')
        ax[1][2].set_xlabel('min')
        ax[1][2].grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print('saved', args.out)


if __name__ == '__main__':
    main()
