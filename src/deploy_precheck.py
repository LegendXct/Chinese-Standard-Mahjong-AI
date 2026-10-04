"""Botzone 部署预检:CPU 单线程端到端延迟 + fp16 存储方案。

1. 延迟:NNPolicy(argmax, CPU, threads=1)四座自打 N 局完整对局(env_fixed),
   统计每个决策点 wall time = 特征构建(request2obs 渲染)+ 前向,
   p50/p95/p99,并给出特征/前向拆分。要求 p99 < 300ms。
2. fp16:权重 half 落盘(加载转回 fp32 推理,PyTorch CPU 的 fp16 计算是软件
   模拟、显著慢于 fp32,fp16 仅作存储格式),报告文件大小、加载转换耗时、
   val 准确率差(要求 <0.1pp,需 GPU)。

用法:
  python src/deploy_precheck.py --ckpt checkpoints/sl_A_gn/best.pt --games 30 --tag A
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_v1 import FeatureAgentV1  # noqa: E402
from env_fixed import MahjongGBEnvFixed  # noqa: E402
from model_v1 import MahjongNetV1, InputDictAdapter  # noqa: E402
from arena.policies import NNPolicy  # noqa: E402
from arena.duplicate import make_wall  # noqa: E402


def measure_latency(ckpt, n_games, seed=777):
    torch.set_num_threads(1)
    net, _ = MahjongNetV1.load(ckpt)
    pol = NNPolicy(InputDictAdapter(net))

    render_times = []
    orig_obs = FeatureAgentV1._obs

    def timed_obs(self):
        t0 = time.perf_counter()
        r = orig_obs(self)
        dt = time.perf_counter() - t0
        self._last_render_dt = dt
        render_times.append(dt)
        return r

    FeatureAgentV1._obs = timed_obs
    lat, fwd = [], []
    try:
        env = MahjongGBEnvFixed(config={'agent_clz': FeatureAgentV1})
        for g in range(n_games):
            obs = env.reset(prevalentWind=g % 4, tileWall=make_wall((seed, g)))
            done = False
            steps = 0
            while not done:
                action_dict = {}
                for name, o in obs.items():
                    seat = int(name.split('_')[1]) - 1
                    t0 = time.perf_counter()
                    action_dict[name] = pol.act(o)
                    dt_f = time.perf_counter() - t0
                    fwd.append(dt_f)
                    lat.append(env.agents[seat]._last_render_dt + dt_f)
                obs, rew, done = env.step(action_dict)
                steps += 1
                assert steps < 1000
    finally:
        FeatureAgentV1._obs = orig_obs

    def q(xs):
        a = np.array(xs) * 1000
        return {'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)),
                'p99': float(np.percentile(a, 99)), 'mean': float(a.mean())}
    return {'n_games': n_games, 'n_decisions': len(lat),
            'latency_ms': q(lat), 'forward_ms': q(fwd), 'render_ms': q(render_times)}


def fp16_check(ckpt, data_root, out_dir):
    model, payload = MahjongNetV1.load(ckpt)
    fp16_path = os.path.join(out_dir, os.path.basename(ckpt).replace('.pt', '_fp16.pt'))
    model.save(fp16_path, half=True, extra={'epoch': payload.get('epoch')})
    size_mb = os.path.getsize(fp16_path) / 2**20
    t0 = time.perf_counter()
    m16, _ = MahjongNetV1.load(fp16_path)  # load 内部已转回 fp32
    load_s = time.perf_counter() - t0

    acc16 = acc32 = None
    if torch.cuda.is_available() and data_root:
        from torch.utils.data import DataLoader
        from dataset_v1 import MahjongV1Dataset, normalize_obs
        ds = MahjongV1Dataset(data_root, 'val', augment=False)
        loader = DataLoader(ds, batch_size=2048, num_workers=8, pin_memory=True)

        @torch.no_grad()
        def acc_of(m):
            m = m.to('cuda').eval()
            n = c = 0
            for obs, mask, act in loader:
                obs = normalize_obs(obs.to('cuda', non_blocking=True))
                mask = mask.to('cuda', non_blocking=True).float()
                act = act.to('cuda', non_blocking=True)
                logits, _ = m(obs)
                pred = (logits + (1 - mask) * (-1e9)).argmax(dim=1)
                c += (pred == act).sum().item()
                n += len(act)
            m.to('cpu')
            return c / n
        acc32 = acc_of(model)
        acc16 = acc_of(m16)
    return {'fp16_path': fp16_path, 'fp16_mb': round(size_mb, 2),
            'load_convert_s': round(load_s, 3),
            'val_acc_fp32': acc32, 'val_acc_fp16': acc16,
            'acc_diff_pp': None if acc32 is None else round((acc32 - acc16) * 100, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--games', type=int, default=30)
    ap.add_argument('--data', default='/root/autodl-tmp/mahjong/data_v1')
    ap.add_argument('--tag', default='model')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print('== latency (%d games, CPU 1 thread) ==' % args.games, flush=True)
    lat = measure_latency(args.ckpt, args.games)
    print(json.dumps(lat, indent=1))
    print('== fp16 storage ==', flush=True)
    fp = fp16_check(args.ckpt, args.data, args.out)
    print(json.dumps(fp, indent=1))
    with open(os.path.join(args.out, 'deploy_precheck_%s.json' % args.tag), 'w') as f:
        json.dump({'ckpt': args.ckpt, 'latency': lat, 'fp16': fp}, f, indent=1)
    ok = lat['latency_ms']['p99'] < 300 and (fp['acc_diff_pp'] is None or abs(fp['acc_diff_pp']) < 0.1)
    print('PRECHECK %s' % ('PASS' if ok else 'FAIL'))


if __name__ == '__main__':
    main()
