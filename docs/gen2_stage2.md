# gen2 阶段二:进攻导向 RL(设计稿，待审定后开训）

状态：**设计稿，未开训**。审定通过后才实现 + 开机。本稿落到 M3 现有 RL 代码
（`src/rl/{config,train_rl,env_worker,infer_server}.py` + `env_fixed.py`）的具体改动点。

## 0. 动机：rl_v1 过拟合的直接证据（阶段一已验证）

阶段一 arena（各 500 组同墙复式）：纯 SL 的 best-v2（`sl_v2_M3_gn_denoise`，83.3%）
**打赢我们的 SL+RL 成品 rl_v1_B +2.250±0.687（>3SE）**，且 best-v2 打 rl_v1_B 比打
sl_champion **赢得更多**、点炮反而更低（14.85% vs 15.65%）。结合 rl_v1 的训练配置
（`reward_scale=30` 纯终局得分、`shaping_c=0` 无塑形、对手池 = 自己快照 + 单一 SL 锚点、
KL 锚定 sl_champion），结论明确：**rl_v1 只在 sl_champion 分布上自对弈，学成了"专打自己人/
专打 SL 同源对手"的策略，对没见过的更强/异质对手脆弱。**

阶段二的两条主线据此设计：**(a) 进攻导向的奖励塑形**（做大牌、合理点炮），
**(b) 对手池强制风格多样化**（杜绝再次过拟合单一分布）。

## 1. 热启动（actor + critic）

- **从 M5 初始化**：`sl_v2_M5_gelu/best.pt`（concat·GELU，阶段一-D 最强基座：val 0.8955、
  arena vs rl_v1_B +7.33±0.73、vs M3′ +4.42±0.75；取代原 M3′/best-v2）。
  这是 **GN** 基座（不依赖 batch 统计，RL 稳定）+ 已训好的 value 头（`val_value≈0.33`），
  actor 与 critic 同源热启动，省 critic 冷启动期。
- **模型从 v1 切到 v2（含 vec）**：rl_v1 用 `MahjongNetV1`；阶段二用 `MahjongNetV2`（双输入 obs+vec）。
  vec 由 `feature_v2.compute_vec_from_obs`（或 C 的 `feature_v2_ext`）**从 obs 现算**——
  arena 的 `NNPolicyV2` 已验证这条路径可跑通。**集成改动点**：
  - `env_worker.py`：rollout 时除 obs 外，同步算 vec 入样本（多一列）。
  - `infer_server.py`：加载 `MahjongNetV2`，前向吃 (obs, vec)；推理批里带 vec。
  - replay_buffer / 样本结构：增加 vec 字段。
  - KL 锚点改为 **M5 本身**（`kl_sl_coef` 复用，锚到起点策略而非 sl_champion）。

## 2. 对手池强制风格多样化（核心修正）

rl_v1 的池 = `['sl'] + 自己快照`（FIFO，`pool_max=20`）。阶段二在此之上**强制注入固定占比、
永不淘汰的偏科陪练**，每个由 best-v2 短训（reward-shaped fine-tune，约 200–400 updates，冻结）。
**异质比例温和引入求稳**：`p_pure_selfplay` 初值 **0.65**（异质 35%，不从 0.75 直接砍到 0.5 以免震荡），
稳定后再降 0.5。对手 3 座按下表**加权采样**：

| 池成员 | 奖励塑形（短训用） | 行为 | 占比 |
|---|---|---|---|
| 自己历史快照 | （主奖励，见 §3） | 当前策略族 | 55% |
| **best-v2 纯基座锚点** | — | 模仿人类分布 | 15% |
| **激进型 aggro** | α=0.6, β=0.0（无视点炮） | 拼大牌、敢点炮 | 10% |
| **防守型 def** | α=0.1, β=0.6（重罚点炮）+ 听牌安全打 | 早弃、安全牌、宁和小不点炮 | 10% |
| **碰碰/混一型 pung** | 碰碰胡/混一色 和牌额外 +0.4 奖励 | 偏做刻子/单一花色 | 5% |
| rl_v1_B（旧成品） | — | "已知会被打的"异质陪练 | 5% |

- 实现：`config.py` 新增 `fixed_sparring: list`（陪练 ckpt + 占比），`train_rl.py` 的
  `ctrl['pool']` 从 `['sl']+pool_ids` 扩成 `固定陪练 + 自己快照`，对手 3 座按上表占比**加权采样**（非均匀）。
- 偏科陪练**不进 FIFO、不被淘汰**（与自己快照分开管理），保证全程都有异质压力。

### 2b. 偏科陪练入池验收（硬门槛）
每个偏科陪练短训后，**入池前**单独 `arena vs best-v2` 测行为指标，确认朝目标方向**显著偏移**才放进池
——避免"假偏科"让多样化名存实亡：
- 激进型：胡牌率 / 胡牌平均番 **显著↑**（vs best-v2 自对弈基线）；
- 防守型：点炮率 **显著↓**；
- 碰碰/混一型：刻子(碰/明杠/暗刻)倾向 / 单花色集中度 **显著↑**。
- 不达标者**不入池**（重训调奖励或弃用）。**验收数据写进报告。**

## 3. 奖励塑形（分批上，先做减法，便于归因）

RL 调试难，三项塑形一起上无法归因。**分两批，逐项叠加验增量**：

### 第一批（先做）：番数正比 α/β 不对称——机理最清晰、invimirage 已验证的核心项

```
r = score/30 + α·(fan/8)·1[自家和牌] − β·(fan/8)·1[自家点炮]
```
- 初值 **α=0.3，β=0.15**（β=α/2，半罚，不恐点炮）；`score/30` 复用 `reward_scale=30`；
  fan 用 `env_fixed` 裁判语义番数（`judge_fan_total`），点炮归属按裁判判定。
- **第一批成功标准（三条全满足才进第二批）**：①胡牌率较 best-v2 自对弈基线**上升**；
  ②点炮率**不破 baseline +3pp 红线**；③固定多样对手集上**泛化均值为正**（见 §4）。
- 红线处置：点炮率破 +3pp → 自动 `α←0.8α、β←1.2β`，连续触发冻结 α；
  策略漂移由 §5 的 KL 锚定（`kl_sl_coef`/`alarm_kl_sl=0.5`）兜底。

### 第二批（第一批确认有效且稳定后，逐个叠加测增量）

- **(a) 番数反向分配（fan backward，参 Tjong/invimirage）**：终局番数奖励沿轨迹反向折扣分配给
  关键早期决策（吃/碰/定向弃牌），让"为做大牌铺垫"的早期动作拿到信用；实现为 advantage 的
  轨迹级再分配（不改 GAE 主体，叠 fan-weighted credit）。**先单独叠这一项，测对 avg_fan/胜率增量。**
- **(b) 向听势能塑形**（potential-based，`shaping_c` 从 0 开）：`r += shaping_c·(γ·Φ(s')−Φ(s))`，
  `Φ=−向听数`；势能差形式不改最优策略。初值 `shaping_c=0.05`，仅早期加速，过拟合即归零。**最后叠。**
- 每叠一项都跑固定对手集复评，**增量为负或训练不稳就回退该项**。

## 4. 防过拟合评测协议（关键）

rl_v1 只快评 `vs 'sl'`。阶段二**评测看泛化，不看自对弈胜率**：

- **固定多样对手集**：`{best-v2, rl_v1_B, RuleBot, 三个偏科陪练}`，每个 `eval_full_groups`
  同墙复式，**报各自配对差 ±SE 与"泛化均值"**（对手集上的平均配对差）。
- **选 eval-best 以泛化均值为准**（而非 vs-sl 单项），杜绝"只会打 SL 同源"再现。
- 额外监控：对**最不擅长的对手**（min 配对差）设地板线，min 掉负且显著则告警。
- `build_eval_tasks` 已支持 `opponent` 参数，扩成对手列表循环即可。

## 5. 复用 M3 基础设施

直接复用、仅按上文调参/扩展：
- **PPO 主体**：GAE `gamma=1.0, gae_lambda=0.95`、`clip=0.15`、`value_coef=0.5`、
  `entropy_coef=0.01`、`grad_clip=1.0`、`ppo_epochs=2`、`rollout_size=65536`。
- **KL 锚定**：`kl_sl_coef=0.05`（锚点换成 best-v2）、`kl_reanchor_max`、`alarm_kl_sl`。
- **对手池管理**：snapshot/FIFO（`pool_snapshot_min`、`pool_max`）+ §2 的固定陪练层。
- **阶梯 lr 控制器 / 首小时门禁 / 停滞看门狗 / 内存哨兵 / infer_server**（扩 v2+vec）。
- **评测/环境**：`env_fixed.MahjongGBEnvFixed`（裁判语义对齐，自摸不双计）。
- lr 初值沿用保守 `2e-5`（热启动微调，防灾难性遗忘）。

## 开训前 checklist（待用户审定）

1. 最终基座 = **M5（`sl_v2_M5_gelu`，concat·GELU）已定**（阶段一-D arena 胜出）；M4(扩特征)、dual(双输入) 均已否决。
2. v2+vec 对 RL 吞吐的影响：vec 现算成本（profiling，必要时 worker 端缓存）。
3. 偏科陪练的短训脚本（3 个 × ~300 updates）需先产出并冻结入 `pool/`。
4. 奖励里 fan 的口径：用 `env_fixed` 的裁判语义番数（`judge_fan_total`），点炮归属判定。
5. 算力预算与 `max_hours`、断点续训、隔离（**绝不动 rl_v1_B 保底物**）。

> 一句话：阶段二不是"再跑一遍 rl_v1"，而是**用进攻奖励把基座往"做大牌"推、用多样对手池把它
> 钉在"打谁都行"上**——直接针对 arena 暴露的 rl_v1 过拟合病根。
