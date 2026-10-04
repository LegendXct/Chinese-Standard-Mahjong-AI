"""集中 GPU 批量推理服务器(1 进程)。

- 常驻 {当前策略 'cur'、SL 锚点 'sl'、对手池快照 'pool_*'、评测副本 'eval_cand'} 的 bf16 副本
- 请求 (worker_id, req_id, model_id, obs uint8, mask uint8, argmax) 经共享 req_q 提交,
  按 model_id 分组攒批(max_batch 或 timeout_ms 先到先服务),前向后经 resp_qs[worker_id] 返回
  (req_id, action, logp, value)
- ctrl_q:('set', model_id, cpu_bf16_sd, version) / ('drop', model_id) / ('stop',)
- 周期性向 stats_q 报 平均批大小 / qps
"""
import queue
import time

import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))


def run_infer_server(cfg, init_models, req_q, resp_qs, ctrl_q, stats_q):
    if cfg.arch == 'v2':
        from model_v2 import MahjongNetV2 as NetClass
        from dataset_v2 import normalize_obs
    else:
        from model_v1 import MahjongNetV1 as NetClass
        from dataset_v1 import normalize_obs
    from rl.common import sample_actions

    torch.set_num_threads(2)
    device = torch.device('cuda')
    models = {}

    def load_model(mid, src):
        if isinstance(src, str):
            net, _ = NetClass.load(src)
        else:
            net, _ = NetClass.load(cfg.sl_champion)
            net.load_state_dict({k: v.float() for k, v in src.items()})
        net = net.to(device, torch.bfloat16).eval()
        models[mid] = net

    for mid, src in init_models.items():
        load_model(mid, src)
    print('[infer] ready with models: %s' % sorted(models), flush=True)

    versions = {}
    n_req = n_batches = 0
    t_stat = time.time()
    pending = []
    timeout_s = cfg.infer_timeout_ms / 1000.0
    tb = {'collect': 0.0, 'forward': 0.0, 'route': 0.0, 'ctrl': 0.0, 'warm': 0.0}
    # GPU 保温:供给瞬时低谷会让 GPU 降频(180MHz),前向变慢→吞吐崩→流量更稀,
    # 形成低位死亡螺旋(实测 25→5 games/s)。空闲超阈值就跑 dummy 前向顶住时钟。
    warm_obs = torch.zeros(16, 102, 4, 9, device=device, dtype=torch.uint8)
    warm_vec = torch.zeros(16, cfg.vec_dim, device=device, dtype=torch.bfloat16)
    t_last_fwd = time.time()

    with torch.no_grad():
        while True:
            _t0 = time.perf_counter()
            try:
                while True:
                    msg = ctrl_q.get_nowait()
                    if msg[0] == 'stop':
                        return
                    if msg[0] == 'set':
                        _, mid, sd, ver = msg
                        if mid in models:
                            models[mid].load_state_dict(
                                {k: v.to(device, torch.bfloat16) for k, v in sd.items()})
                        else:
                            load_model(mid, sd)
                        versions[mid] = ver
                    elif msg[0] == 'drop':
                        models.pop(msg[1], None)
            except queue.Empty:
                pass
            tb['ctrl'] += time.perf_counter() - _t0

            # 攒批
            _t0 = time.perf_counter()
            t_first = None
            while len(pending) < cfg.infer_max_batch:
                try:
                    r = req_q.get(timeout=0.0005 if pending else 0.005)
                    pending.append(r)
                    if t_first is None:
                        t_first = time.time()
                except queue.Empty:
                    if pending:
                        break
                    # 空闲保温:发射间隙 >8ms 会让 GPU 降频(实测 batch12 前向
                    # 2.75ms -> 9.4ms),触发吞吐死亡螺旋,用 dummy 前向顶住时钟
                    if time.time() - t_last_fwd > 0.008:
                        _tw = time.perf_counter()
                        models['cur'](normalize_obs(warm_obs).to(torch.bfloat16), warm_vec)
                        torch.cuda.synchronize()
                        t_last_fwd = time.time()
                        tb['warm'] += time.perf_counter() - _tw
                    t_first = None
                    continue
                if t_first and time.time() - t_first >= timeout_s:
                    break
            tb['collect'] += time.perf_counter() - _t0
            if not pending:
                continue

            batch, pending = pending, []
            by_model = {}
            for r in batch:
                by_model.setdefault((r[2], r[6]), []).append(r)
            _t0 = time.perf_counter()
            outs = []
            for (mid, argmax), rs in by_model.items():
                net = models.get(mid)
                if net is None:
                    # 池淘汰('drop')与在飞请求存在竞态:跨越淘汰时刻的对局
                    # 仍会按旧 model_id 发请求。回落到常驻锚点,绝不让服务进程崩
                    # (2026-06-11 实测 KeyError pool_00045 冻结整个主跑 75min)
                    net = models.get('sl') or models['cur']
                    print('[infer] WARN dropped model %s requested, fallback' % mid,
                          flush=True)
                obs = torch.from_numpy(np.stack([r[3] for r in rs])).to(device)
                vec = torch.from_numpy(np.stack([r[4] for r in rs])).to(device, torch.bfloat16)
                mask = torch.from_numpy(np.stack([r[5] for r in rs])).to(device)
                obs_f = normalize_obs(obs).to(torch.bfloat16)
                logits, value = net(obs_f, vec)
                logits = logits.float()
                acts, logp = sample_actions(logits, mask.float(), argmax=argmax)
                outs.append((rs, acts.cpu().numpy(), logp.cpu().numpy(),
                             value.float().cpu().numpy()))
            tb['forward'] += time.perf_counter() - _t0
            t_last_fwd = time.time()
            _t0 = time.perf_counter()
            for rs, acts, logp, value in outs:
                for i, r in enumerate(rs):
                    resp_qs[r[0]].put((r[1], int(acts[i]), float(logp[i]), float(value[i])))
            tb['route'] += time.perf_counter() - _t0
            n_req += len(batch)
            n_batches += 1
            if time.time() - t_stat > 30:
                stats_q.put(('infer', {'qps': round(n_req / (time.time() - t_stat), 1),
                                       'avg_batch': round(n_req / max(n_batches, 1), 1),
                                       'models': len(models),
                                       'tb': {k: round(v, 1) for k, v in tb.items()}}))
                n_req = n_batches = 0
                tb = {k: 0.0 for k in tb}
                t_stat = time.time()
