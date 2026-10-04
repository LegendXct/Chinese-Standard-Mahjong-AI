"""单进程同步 PPO 的核心数值:masked 分布工具 + 终局型 GAE + KL 锚定 PPO 更新。

数学与 src/rl/common.py(旧异步框架)同源(masked log_softmax / dist_stats / kl_masked),
但本文件自包含、不依赖那套异步包,专为 src/rl/sync 的单进程同步训练写。
模型 = MahjongNetV2(obs+vec 双输入,策略/价值双头)。
"""
import numpy as np
import torch
import torch.nn.functional as F

NEG = -1e9


def masked_log_softmax(logits, mask):
    """logits (B,235), mask (B,235) in {0,1} -> 合法动作上的 log 概率(非法位压到 -inf 级)。"""
    return F.log_softmax(logits + (1.0 - mask) * NEG, dim=-1)


def dist_stats(logits, mask, actions=None):
    """返回 (logp_actions|None, entropy, logp_all);熵只在合法动作分布上算。"""
    logp_all = masked_log_softmax(logits, mask)
    p = logp_all.exp()
    entropy = -(p * logp_all.clamp(min=NEG)).sum(-1)
    logp_a = None
    if actions is not None:
        logp_a = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return logp_a, entropy, logp_all


def kl_masked(logp_p, logp_q, mask):
    """KL(P‖Q),两边均为 masked log_softmax 输出;非法位 P=0 不贡献。"""
    p = logp_p.exp()
    return (p * (logp_p - logp_q) * mask).sum(-1)


def compute_gae(values, final_reward, gamma, lam):
    """终局奖励型 GAE:仅最后一步 reward=final_reward,中间步 reward=0,尾部 bootstrap=0。

    values: 1D np.float32 (T,) —— 收集时记录的 critic 估值。返回 (adv, ret) 均 (T,) np.float32。
    """
    T = len(values)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        v_next = values[t + 1] if t + 1 < T else 0.0
        r = final_reward if t == T - 1 else 0.0
        delta = r + gamma * v_next - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def compute_gae_rewards(values, rewards, gamma, lam):
    """逐步奖励型 GAE:rewards 为长度 T 的逐步奖励数组(非仅终局),尾部 bootstrap=0。

    Step-2 的 terminal 模式 == 本函数当 rewards=[0,...,0,R];fanback / potential 模式用
    密集 rewards。返回 (adv, ret),ret = adv + values 为与逐步奖励自洽的 GAE return-to-go
    (= 价值头回归目标,保证 advantage 与 value target 同口径)。
    """
    T = len(values)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        v_next = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * v_next - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def ppo_update(model, frozen_model, opt, batch, cfg, device, normalize_obs):
    """一次 PPO 更新(cfg.ppo_epochs 轮 minibatch 遍历)。

    batch: dict of np 数组,字段对齐 selfplay 产出:
        obs(uint8 N,102,4,9), vec(f32 N,107), mask(u8 N,235),
        act(i64 N), logp(f32 N), adv(f32 N), ret(f32 N)。
    KL 锚定项 = cfg.beta_kl * KL(π_frozen_M5 ‖ π_θ)(任务指定的前向 KL,锚点权重 = 冻结 M5 概率)。
    优势在整个 batch 上标准化。返回标量统计 dict(全 float / int)。
    """
    obs = torch.from_numpy(batch['obs'])
    vec = torch.from_numpy(batch['vec'])
    mask = torch.from_numpy(batch['mask'])
    act = torch.from_numpy(batch['act'])
    logp_old = torch.from_numpy(batch['logp'])
    adv_raw = torch.from_numpy(batch['adv'])
    ret = torch.from_numpy(batch['ret'])
    N = len(act)
    mb = min(cfg.minibatch, N)

    adv_norm = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

    keys = ('policy', 'value', 'entropy', 'kl_anchor', 'kl_rev',
            'approx_kl', 'clip_frac', 'ratio_mean')
    acc = {k: 0.0 for k in keys}
    nmb = 0
    gn = 0.0
    model.train()
    for _ep in range(cfg.ppo_epochs):
        perm = torch.randperm(N)
        for i0 in range(0, N, mb):
            idx = perm[i0:i0 + mb]
            if len(idx) < 2:                      # 末尾极小 minibatch 跳过(adv 标准化无意义)
                continue
            o = normalize_obs(obs[idx].to(device))
            vc = vec[idx].to(device)
            m = mask[idx].to(device).float()
            a = act[idx].to(device)
            lp_old = logp_old[idx].to(device)
            A = adv_norm[idx].to(device)
            R = ret[idx].to(device)

            logits, value = model(o, vc)
            with torch.no_grad():
                f_logits, _ = frozen_model(o, vc)
            lp_new, ent, logp_all = dist_stats(logits, m, a)
            _, _, logp_frozen = dist_stats(f_logits, m)

            ratio = (lp_new - lp_old).exp()
            pl = -torch.min(ratio * A,
                            ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
            vl = F.smooth_l1_loss(value, R)
            ent_mean = ent.mean()
            kl_anchor = kl_masked(logp_frozen, logp_all, m).mean()   # KL(frozen ‖ θ)
            loss = pl + cfg.value_coef * vl - cfg.entropy_coef * ent_mean \
                + cfg.beta_kl * kl_anchor

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            with torch.no_grad():
                acc['policy'] += pl.item()
                acc['value'] += vl.item()
                acc['entropy'] += ent_mean.item()
                acc['kl_anchor'] += kl_anchor.item()
                acc['kl_rev'] += kl_masked(logp_all, logp_frozen, m).mean().item()
                acc['approx_kl'] += (lp_old - lp_new).mean().item()   # 近似 KL(π_old_θ ‖ π_new_θ)
                acc['clip_frac'] += ((ratio - 1.0).abs() > cfg.clip).float().mean().item()
                acc['ratio_mean'] += ratio.mean().item()
            nmb += 1

    out = {k: v / max(nmb, 1) for k, v in acc.items()}
    out['n_minibatches'] = nmb
    out['grad_norm_last'] = float(gn)
    out['n_samples'] = int(N)
    return out
