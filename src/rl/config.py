"""gen2 阶段二 PPO 自对弈配置(全部可在 CLI 用 --set key=value 覆盖)。

阶段二 = v2 双输入(obs+vec)+ 番数正比奖励塑形 + 校准版多样对手池。
基座 M5(sl_v2_M5_gelu),新 run_dir rl_v2,绝不动 rl_v1 保底物。
"""
from dataclasses import dataclass


@dataclass
class RLConfig:
    # 路径(stage-2:M5 热启动;run_dir 独立,绝不动 rl_v1)
    sl_champion: str = '/root/autodl-tmp/mahjong/checkpoints/sl_v2_M5_gelu/best.pt'
    init_model: str = ''              # 非空时 actor/critic 从它热启动(KL 锚点仍 = sl_champion=M5)
    run_dir: str = '/root/autodl-tmp/mahjong/checkpoints/rl_v2'
    data_root: str = '/root/autodl-tmp/mahjong/data_v1'

    # 架构(v2 双输入;vec 由 feature_v2.compute_vec_from_obs 现算)
    arch: str = 'v2'
    vec_dim: int = 107
    n_env_workers: int = 18
    envs_per_worker: int = 8
    infer_max_batch: int = 512
    infer_timeout_ms: float = 4.0

    # 对局模式(校准版对手池;p_pure_selfplay 初值 0.65,异质 35%)
    p_pure_selfplay: float = 0.65
    pool_max: int = 20                # 自己快照 FIFO 上限(不含固定陪练)
    pool_snapshot_min: float = 30.0
    # mixed 局 3 座按权重加权采样(权重无需 sum=100,内部归一)
    w_self_snap: float = 55.0         # 自己历史快照
    w_anchor: float = 15.0            # M5 锚点('sl' 模型)
    w_bighand: float = 13.0           # 高番做大牌型陪练
    w_def: float = 12.0               # 防守型陪练
    w_rl_v1b: float = 5.0             # rl_v1_B 异质陪练
    spar_dir: str = '/root/autodl-tmp/mahjong/checkpoints/rl_v2_spar'
    rl_v1b_ckpt: str = '/root/autodl-tmp/mahjong/checkpoints/rl_v1/last.pt'

    # 奖励塑形(第一批:番数正比 α/β 不对称)/ GAE
    reward_scale: float = 30.0
    reward_alpha: float = 0.3         # 自家和牌 r += α·(fan/8)^pow
    reward_beta: float = 0.15         # 自家点炮 r -= β·(fan/8)
    reward_fan_thresh: int = 0        # 仅 fan>=thresh 的和牌给番奖励(注:国标起胡=8番,<8 无效)
    reward_fan_pow: float = 1.0       # 番奖励指数:>1 非线性放大高番(做大牌陪练用,如 2.0)
    reward_fan_cap: int = 0           # 番奖励封顶 min(fan,cap),0=不封(抑制大牌方差)
    gamma: float = 1.0
    gae_lambda: float = 0.95
    shaping_c: float = 0.0            # 第二批向听势能塑形,本批关闭
    # fan backward(第二批进攻信用分配:终局番沿和牌方轨迹反向折扣加到 advantage,只动 adv 不动 ret)
    fan_backward_coef: float = 0.0    # c_fb;>0 启用(叠加 GAE 不替代);v1 用 0.3
    fan_backward_gamma: float = 0.9   # γ_fb 反向折扣(半衰 ~7 步)
    fan_backward_min_fan: int = 12    # 只对 fan>=此值的和牌反传(集中大牌通路)
    fan_backward_fan_cap: int = 24    # g(fan)=(min(fan,cap)/8)^2 封顶控方差
    # 防守:番数敏感点炮罚 r -= β·(min(fan_opp,cap)/8)^dealin_pow(p_def>1:点大番远重于小番)
    dealin_pow: float = 1.0           # 1.0=线性(向后兼容);第二批 2.0
    dealin_fan_cap: int = 0           # 对方番封顶(0=不封);第二批 24
    # value warmup:fan backward 抬 advantage 方差,前 N upd value_coef ×mult 先练好 value head
    value_warmup_updates: int = 0
    value_warmup_mult: float = 1.0

    # PPO
    rollout_size: int = 65536
    minibatch: int = 8192
    ppo_epochs: int = 2
    clip: float = 0.15
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_sl_coef: float = 0.05          # β·KL(π‖π_M5) 锚定(锚点=M5)
    lr: float = 2e-5
    wd: float = 1e-4
    grad_clip: float = 1.0

    # 运行控制
    seed: int = 20260610
    eval_every_min: float = 30.0      # 快评节奏(vs M5,看趋势)
    eval_groups: int = 200
    eval_full_min: float = 120.0      # 完整(泛化)评测节奏
    eval_full_groups: int = 500
    eval_floor_se: float = 1.0        # eval-best 地板:comp_min >= -eval_floor_se×SE 才更新(收紧用 0.5)
    n_eval_workers: int = 4
    kl_reanchor_max: int = 2
    first_hour_gate: bool = False
    log_every_min: float = 15.0
    ckpt_every_updates: int = 10
    max_hours: float = 4.0
    max_updates: int = 0              # >0 时按更新数停(偏科陪练短训用)
    # 告警线
    alarm_entropy_frac: float = 0.40
    alarm_kl_sl: float = 0.5
    alarm_hu_drop: float = 0.30
    alarm_thru_drop: float = 0.5
    alarm_dianpao_pp: float = 0.03    # 点炮率破 baseline +3pp 红线 -> 压 α 升 β


def parse_overrides(cfg, pairs):
    for p in pairs or []:
        k, v = p.split('=', 1)
        old = getattr(cfg, k)
        if isinstance(old, bool):
            setattr(cfg, k, v.lower() in ('1', 'true', 'yes'))
        elif isinstance(old, int):
            setattr(cfg, k, int(v))
        elif isinstance(old, float):
            setattr(cfg, k, float(v))
        else:
            setattr(cfg, k, v)
    return cfg
