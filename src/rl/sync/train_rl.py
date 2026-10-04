"""单进程同步 PPO 微调主程序(KL 锚定到 M5)。

设计:N 个 CPU 自对弈 worker(spawn Pool)收集整局 episode -> 主进程在 GPU 做 PPO 更新
-> 下一轮注入新权重再收集。actor 与 critic 同一 MahjongNetV2,从 M5 热启动;另常驻一份
冻结 M5 作 KL 锚点(总 loss = PPO_clip + value_coef·value + (-entropy_coef·H)
                         + beta_kl·KL(π_M5 ‖ π_θ))。
对手 = 3 家冻结 M5(默认 argmax,口径同 arena 1v3,目标"打赢 M5")。

不依赖、不改动 src/rl 旧异步框架(infer_server/env_worker/共享内存)。

--smoke:极小规模验证管线正确性与吞吐,不做正式训练。例:
  python src/rl/sync/train_rl.py --smoke --workers 8 --games-per-round 120 --rounds 2 \
      --out-dir /root/autodl-tmp/mahjong/checkpoints/rl_smoke
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, '..', '..'))
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_v2 import MahjongNetV2          # noqa: E402
from dataset_v2 import normalize_obs       # noqa: E402
import ppo as ppo_mod                       # noqa: E402
import selfplay as sp                       # noqa: E402


@dataclass
class SyncConfig:
    m5_path: str = '/root/autodl-tmp/mahjong/checkpoints/sl_v2_M5_gelu/best.pt'
    out_dir: str = '/root/autodl-tmp/mahjong/checkpoints/rl_smoke'
    workers: int = 8
    games_per_round: int = 120
    rounds: int = 2
    # reward / GAE
    reward_scale: float = 30.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    opp_sample: bool = False        # 对手冻结 M5:False=argmax(同 arena),True=温度1采样
    # PPO
    minibatch: int = 2048
    ppo_epochs: int = 2
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    beta_kl: float = 0.02           # 冒烟用小值,先验证管线
    lr: float = 2e-5
    wd: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 20260610
    snapshots: int = 0              # 训练中等距存的中间快照数(不含末尾 last.pt);如 2 -> 在 1/3,2/3 处
    # 奖励/信用分配模式(Step 3;terminal 为 Step 2 行为,不被破坏)
    reward_mode: str = 'terminal'   # terminal | fanback | potential | combo(fanback⊕potential)
    fanback_gamma: float = 0.99     # fanback:终局 R 沿轨迹反向折扣权重的衰减(0.99≈近均匀,credit 含早期布局)
    potential_coef: float = 0.1     # potential:势能塑形 c·F 的小系数(远小于终局 R,避免淹没)


def _cast(old, v):
    if isinstance(old, bool):
        return v.lower() in ('1', 'true', 'yes')
    if isinstance(old, int):
        return int(v)
    if isinstance(old, float):
        return float(v)
    return v


def _rss_bytes(pid):
    try:
        with open('/proc/%d/statm' % pid) as f:
            pages = int(f.read().split()[1])     # resident pages
        return pages * os.sysconf('SC_PAGE_SIZE')
    except Exception:
        return 0


def tree_rss_gb(child_pids):
    total = _rss_bytes(os.getpid()) + sum(_rss_bytes(p) for p in child_pids)
    return total / (1024 ** 3)


def gpu_probe():
    """返回 (util%, mem_used_MB) via nvidia-smi;失败给 (-1,-1)。"""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'], timeout=5).decode().strip().splitlines()[0]
        u, m = out.split(',')
        return int(u), int(m)
    except Exception:
        return -1, -1


def serialize_sd(model):
    """state_dict -> 单个 bytes 块。逐 tensor 经 Pool 传递会触发 torch 共享内存 FD 化
    (8 worker × ~150 tensor 撑爆 ulimit -n=1024 -> 'Too many open files' 卡死);
    打包成一个 bytes(plain pickle,零 FD),worker 端 torch.load 还原。"""
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return buf.getvalue()


def raise_fd_limit():
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        return hard
    except Exception:
        return -1


def build_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for _n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': wd},
         {'params': no_decay, 'weight_decay': 0.0}], lr=lr)


def save_ckpt(model, path, meta):
    """存 {config, state_dict(fp32 cpu), rl:meta} —— arena nnv2: / 部署 FP16ResidentNetV2 直接可读。"""
    sd_cpu = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    tmp = path + '.tmp'
    torch.save({'config': model.config, 'state_dict': sd_cpu, 'rl': meta}, tmp)
    os.replace(tmp, path)


def merge_stats(results):
    agg = {'games': 0, 'samples': 0, 'wins': 0, 'huang': 0, 'illegal': 0,
           'mask_assert_fail': 0, 'shap_sum': 0.0, 'shap_absmean': 0.0, 'shap_n': 0,
           'final_scores': [], 'fans': [], 'steps': []}
    trajs = []
    for tj, st in results:
        trajs.extend(tj)
        for k in ('games', 'samples', 'wins', 'huang', 'illegal', 'mask_assert_fail',
                  'shap_sum', 'shap_absmean', 'shap_n'):
            agg[k] += st[k]
        for k in ('final_scores', 'fans', 'steps'):
            agg[k].extend(st[k])
    return trajs, agg


def concat_batch(trajs):
    keys = ('obs', 'vec', 'mask', 'act', 'logp', 'adv', 'ret')
    return {k: np.concatenate([t[k] for t in trajs], axis=0) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int)
    ap.add_argument('--games-per-round', type=int)
    ap.add_argument('--rounds', type=int)
    ap.add_argument('--out-dir')
    ap.add_argument('--beta-kl', type=float)
    ap.add_argument('--minibatch', type=int)
    ap.add_argument('--snapshots', type=int)
    ap.add_argument('--reward-mode', choices=['terminal', 'fanback', 'potential', 'combo'])
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()

    cfg = SyncConfig()
    for pair in args.set or []:
        k, v = pair.split('=', 1)
        setattr(cfg, k, _cast(getattr(cfg, k), v))
    for k_arg, k_cfg in (('workers', 'workers'), ('games_per_round', 'games_per_round'),
                         ('rounds', 'rounds'), ('out_dir', 'out_dir'),
                         ('beta_kl', 'beta_kl'), ('minibatch', 'minibatch'),
                         ('snapshots', 'snapshots'), ('reward_mode', 'reward_mode')):
        v = getattr(args, k_arg)
        if v is not None:
            setattr(cfg, k_cfg, v)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device('cuda')

    # actor-critic θ(从 M5 热启动)+ 冻结 M5 锚点
    model, payload = MahjongNetV2.load(cfg.m5_path, map_location='cpu')
    model = model.to(device)
    frozen, _ = MahjongNetV2.load(cfg.m5_path, map_location='cpu')
    frozen = frozen.to(device).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    opt = build_optimizer(model, cfg.lr, cfg.wd)

    print('=== sync PPO %s ===' % ('SMOKE' if args.smoke else 'TRAIN'), flush=True)
    print('config: %s' % json.dumps(asdict(cfg)), flush=True)
    print('model: config=%s params_M=%.4f  (M5 val_acc=%s)' % (
        model.config, model.n_params() / 1e6, payload.get('val_acc')), flush=True)

    fd_hard = raise_fd_limit()
    print('fd_limit raised to hard=%s' % fd_hard, flush=True)
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(cfg.workers, initializer=sp.worker_init, initargs=(cfg.m5_path,))
    try:
        child_pids = [p.pid for p in pool._pool]
    except Exception:
        child_pids = []

    summary = {'rounds': []}
    gpu_peak_mb = 0
    rss_peak_gb = 0.0
    cum_games = cum_samples = 0
    snap_rounds = set()
    if cfg.snapshots > 0:
        for k in range(1, cfg.snapshots + 1):
            snap_rounds.add(max(1, round(cfg.rounds * k / (cfg.snapshots + 1))))
    print('snapshot at completed-rounds: %s ; final -> %s' % (
        sorted(snap_rounds), 'smoke.pt' if args.smoke else 'last.pt'), flush=True)
    try:
        for r in range(cfg.rounds):
            cur_blob = serialize_sd(model)        # 单 bytes 块,避免逐 tensor FD 化卡死
            gpw = (cfg.games_per_round + cfg.workers - 1) // cfg.workers   # 每 worker 上取整
            tasks = [(cur_blob, gpw, cfg.seed, r, w, cfg.gamma, cfg.gae_lambda,
                      cfg.reward_scale, cfg.opp_sample,
                      cfg.reward_mode, cfg.fanback_gamma, cfg.potential_coef)
                     for w in range(cfg.workers)]

            t0 = time.time()
            results = pool.starmap(sp.collect, tasks)
            t_collect = time.time() - t0
            trajs, agg = merge_stats(results)

            if not trajs:
                print('[round %d] WARN: 0 trajectories collected, skip update' % r, flush=True)
                continue
            batch = concat_batch(trajs)
            n_samp = len(batch['act'])
            cum_games += agg['games']
            cum_samples += n_samp
            fs = np.array(agg['final_scores'], dtype=np.float32)

            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            st = ppo_mod.ppo_update(model, frozen, opt, batch, cfg, device, normalize_obs)
            torch.cuda.synchronize()
            t_update = time.time() - t1

            gpu_alloc_mb = torch.cuda.max_memory_allocated() / 1e6
            gpu_resv_mb = torch.cuda.max_memory_reserved() / 1e6
            gpu_util, gpu_used_mb = gpu_probe()
            rss_gb = tree_rss_gb(child_pids)
            gpu_peak_mb = max(gpu_peak_mb, gpu_resv_mb)
            rss_peak_gb = max(rss_peak_gb, rss_gb)

            finite_keys = ('policy', 'value', 'entropy', 'kl_anchor', 'kl_rev',
                           'approx_kl', 'clip_frac', 'ratio_mean')
            finite_ok = all(np.isfinite(st[k]) for k in finite_keys)

            rec = {
                'round': r, 'games': agg['games'], 'samples': n_samp,
                'cum_games': cum_games, 'cum_samples': cum_samples,
                'collect_s': round(t_collect, 1), 'update_s': round(t_update, 2),
                'games_per_s': round(agg['games'] / t_collect, 2),
                'samples_per_s': round(n_samp / t_collect, 1),
                'win_rate': round(agg['wins'] / max(agg['games'], 1), 4),
                'huang_rate': round(agg['huang'] / max(agg['games'], 1), 4),
                'avg_fan': round(float(np.mean(agg['fans'])) if agg['fans'] else 0.0, 2),
                'train_seat_score_mean': round(float(fs.mean()), 3),
                'mean_steps': round(float(np.mean(agg['steps'])), 1),
                'illegal': agg['illegal'], 'mask_assert_fail': agg['mask_assert_fail'],
                'reward_mode': cfg.reward_mode,
                'shap_absmean': round(agg['shap_absmean'] / max(agg['shap_n'], 1), 5),
                'reward_absmean': round(float(np.abs(fs).mean()) / cfg.reward_scale, 5),
                'policy_loss': round(st['policy'], 5), 'value_loss': round(st['value'], 5),
                'entropy': round(st['entropy'], 5), 'kl_anchor': st['kl_anchor'],
                'kl_rev': st['kl_rev'], 'approx_kl': st['approx_kl'],
                'clip_frac': round(st['clip_frac'], 5), 'ratio_mean': round(st['ratio_mean'], 5),
                'grad_norm': round(st['grad_norm_last'], 4), 'n_minibatches': st['n_minibatches'],
                'finite_ok': finite_ok,
                'gpu_util_pct': gpu_util, 'gpu_used_mb': gpu_used_mb,
                'gpu_peak_alloc_mb': round(gpu_alloc_mb, 1),
                'gpu_peak_resv_mb': round(gpu_resv_mb, 1), 'tree_rss_gb': round(rss_gb, 2),
            }
            summary['rounds'].append(rec)
            print('[round %d] %s' % (r, json.dumps(rec, ensure_ascii=False)), flush=True)

            completed = r + 1
            if completed in snap_rounds:
                sp_path = os.path.join(cfg.out_dir, 'snap_%04d.pt' % completed)
                save_ckpt(model, sp_path, {'round': completed, 'cum_games': cum_games,
                                           'cum_samples': cum_samples, 'beta_kl': cfg.beta_kl,
                                           'rec': rec})
                print('[SNAPSHOT] %s after %d rounds (%d games, %d samples)' % (
                    sp_path, completed, cum_games, cum_samples), flush=True)
    finally:
        pool.close()
        pool.join()

    # 末尾 checkpoint(config + state_dict,arena nnv2: / 部署直接可读)
    final_name = 'smoke.pt' if args.smoke else 'last.pt'
    ckpt = os.path.join(cfg.out_dir, final_name)
    save_ckpt(model, ckpt, {'final': True, 'cum_games': cum_games, 'cum_samples': cum_samples,
                            'config': asdict(cfg), 'rounds': summary['rounds']})

    all_illegal = sum(rec['illegal'] for rec in summary['rounds'])
    all_finite = all(rec['finite_ok'] for rec in summary['rounds'])
    # 棋力/稳定性轨迹(便于 grep 取曲线):每轮 score / entropy / kl_anchor / clip_frac
    print('TRAJECTORY round,cum_games,train_score,entropy,kl_anchor,kl_rev,clip_frac,illegal', flush=True)
    for rec in summary['rounds']:
        print('TRAJ %d,%d,%.3f,%.4f,%.6f,%.6f,%.5f,%d' % (
            rec['round'], rec['cum_games'], rec['train_seat_score_mean'], rec['entropy'],
            rec['kl_anchor'], rec['kl_rev'], rec['clip_frac'], rec['illegal']), flush=True)
    print('%s %s' % ('SMOKE_SUMMARY' if args.smoke else 'TRAIN_SUMMARY', json.dumps({
        'checkpoint': ckpt, 'rounds': len(summary['rounds']),
        'cum_games': cum_games, 'cum_samples': cum_samples,
        'beta_kl': cfg.beta_kl, 'lr': cfg.lr,
        'total_illegal': all_illegal, 'all_finite': all_finite,
        'kl_anchor_first': summary['rounds'][0]['kl_anchor'] if summary['rounds'] else None,
        'kl_anchor_last': summary['rounds'][-1]['kl_anchor'] if summary['rounds'] else None,
        'entropy_first': summary['rounds'][0]['entropy'] if summary['rounds'] else None,
        'entropy_last': summary['rounds'][-1]['entropy'] if summary['rounds'] else None,
        'gpu_peak_resv_mb': round(gpu_peak_mb, 1), 'tree_rss_peak_gb': round(rss_peak_gb, 2),
    }, ensure_ascii=False)), flush=True)
    print('SMOKE_DONE' if args.smoke else 'TRAIN_DONE')


if __name__ == '__main__':
    main()
