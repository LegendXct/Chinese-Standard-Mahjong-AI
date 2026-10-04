"""actor 与 learner 共用的分布工具:masked log_prob / 熵 / 采样。

唯一实现,杜绝两处漂移:mask 加 -1e9 后 log_softmax。
"""
import torch
import torch.nn.functional as F

NEG = -1e9


def masked_log_softmax(logits, mask):
    """logits (B,235) float, mask (B,235) float{0,1} -> 合法动作上的 log 概率。"""
    return F.log_softmax(logits + (1.0 - mask) * NEG, dim=-1)


def dist_stats(logits, mask, actions=None):
    """返回 (logp_actions, entropy, logp_all)。熵只在合法动作分布上计算。"""
    logp_all = masked_log_softmax(logits, mask)
    p = logp_all.exp()
    entropy = -(p * logp_all.clamp(min=NEG)).sum(-1)
    logp_a = None
    if actions is not None:
        logp_a = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return logp_a, entropy, logp_all


def sample_actions(logits, mask, argmax=False):
    """温度 1 采样(或 argmax)。返回 (actions, logp)。"""
    logp_all = masked_log_softmax(logits, mask)
    if argmax:
        actions = logp_all.argmax(-1)
    else:
        actions = torch.multinomial(logp_all.exp(), 1).squeeze(-1)
    logp = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return actions, logp


def kl_masked(logp_p, logp_q, mask):
    """KL(P‖Q),两者均为 masked log_softmax 输出;非法位概率为 0 不贡献。"""
    p = logp_p.exp()
    return (p * (logp_p - logp_q) * mask).sum(-1)


def sd_to_cpu_bf16(state_dict):
    return {k: (v.detach().to('cpu', torch.bfloat16) if v.is_floating_point()
                else v.detach().cpu()) for k, v in state_dict.items()}


# ---------------- 轨迹共享内存 arena ----------------
# 大轨迹(~60KB+)走 mp.Queue 会被生产者 feeder 线程的 GIL 饥饿卡死
# (worker 主线程热循环持有 GIL,实测读端只剩 ~20 件/s)。
# 改为:worker 把轨迹数组写进共享内存槽位(numpy memcpy 释放 GIL),
# 队列只传 (worker_id, slot, T) 小描述符;槽位用每 worker 的信号量做环形配额,
# 单生产者 FIFO 保证消费顺序 == 生产顺序,round-robin 槽位安全。
import numpy as _np

TRAJ_SLOTS = 64                 # 每 worker 槽位数(64×512KB×23 worker ≈ 754MB 共享内存)
TRAJ_SLOT_BYTES = 1 << 19       # 512KB:v2 单步 ~4.3KB,容纳 ~120 步,实际单座 <60 步
_VEC_DIM = 107                  # 与 feature_v2.VEC_DIM 同步(stage-2 双输入 vec)
_STEP_FIELDS = (('obs', _np.uint8, 102 * 4 * 9), ('vec', _np.float32, _VEC_DIM),
                ('mask', _np.uint8, 235),
                ('act', _np.int64, 1), ('logp', _np.float32, 1),
                ('adv', _np.float32, 1), ('ret', _np.float32, 1))


def traj_capacity():
    step_bytes = sum(_np.dtype(dt).itemsize * n for _, dt, n in _STEP_FIELDS)
    return TRAJ_SLOT_BYTES // step_bytes


def write_traj(slot_view, traj):
    """traj: dict of arrays(与 env_worker 产出一致)。返回 T。"""
    T = len(traj['act'])
    off = 0
    for name, dt, n in _STEP_FIELDS:
        a = _np.ascontiguousarray(traj[name], dtype=dt).reshape(T, -1)
        nb = a.nbytes
        slot_view[off:off + nb] = a.view(_np.uint8).reshape(-1)
        off += nb
    return T


def read_traj(slot_view, T):
    out = {}
    off = 0
    for name, dt, n in _STEP_FIELDS:
        nb = T * n * _np.dtype(dt).itemsize
        out[name] = _np.frombuffer(bytes(slot_view[off:off + nb]), dtype=dt).reshape(T, n)
        if n == 1:
            out[name] = out[name].reshape(T)
        off += nb
    out['obs'] = out['obs'].reshape(T, 102, 4, 9)
    return out
