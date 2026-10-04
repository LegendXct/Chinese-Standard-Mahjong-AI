# Gen2 设计稿:完美信息 Critic + 风格化对手池

状态:预研(6/13);GPU 训练待 Round 1 结果(6/20)后决定。
配套实验:`scripts/oracle_gate.py`(oracle 价值头离线 gate,CPU 可跑)。

---

## 一、完美信息 Critic(CTDE:训练时全知,部署时不变)

### 1.1 动机(M3 实测证据)

M3.2 的 value loss 全程平台(0.16~0.18 无下降趋势),而 policy 持续改善——
不完美信息下"这手牌最终值多少分"本质上不可知(对手暗牌/牌墙未知),critic
回归的是高方差期望,GAE 优势估计的噪声直接限制策略梯度质量。训练期让 critic
看全局(对手暗牌、牌墙),policy 塔保持 102 通道不变,部署零影响——标准 CTDE。

### 1.2 Oracle 特征规格(critic 侧附加 28 通道,总 130)

| 通道 | 内容 | 编码 |
|---|---|---|
| 0-3 | 下家暗手牌 | 34 牌种 4×9 网格,计数温度计(≥1/≥2/≥3/≥4)|
| 4-7 | 对家暗手牌 | 同上 |
| 8-11 | 上家暗手牌 | 同上 |
| 12-15 | 牌墙剩余构成 | 每牌种剩余张数温度计(墙=全 136 − 已发 − 已摸,从牌谱/env 精确可知)|
| 16-18 | 自家向听距离 | sh≤0 / ≤1 / ≤2 全平面温度计 |
| 19-21 | 下家向听距离 | 同上 |
| 22-24 | 对家向听距离 | 同上 |
| 25-27 | 上家向听距离 | 同上 |

设计要点:座位用相对序(自/下/对/上)与现有 102 通道的 packs 相对序一致;
全部 0/1 平面,无需新归一化;向听用 shanten_py(部署不含此路径,无兼容负担)。
备选扩展(二期再议):各家有效进张数标量面、立直型听牌张位图。

### 1.3 网络结构:双塔 vs 共享主干+侧支

| 方案 | 结构 | 参数量 | 训练成本 | 部署影响 |
|---|---|---|---|---|
| A 双塔独立 | policy 塔 = 现 MahjongNetV1(102);critic 塔独立小网(130 入,W128/d10)| +~3.5M | 前向 ×~1.5(critic 塔只在训练侧跑)| **零**(部署只导 policy 塔)|
| B 共享主干+oracle 侧支 | 主干吃 102;oracle 28 经 1×1 conv 嵌入后在第 k 层注入 critic 分支 | +~0.5M | 共享省算力,但**主干梯度被 value 损失污染**:oracle 信息经反传渗入 policy 表征,部署时该表征失去 oracle 输入会分布漂移 | 零(结构上),但有隐性风险 |

**推荐:A 双塔独立。** 论据:(1) B 的渗入风险正是 CTDE 文献反复警告的
(centralized critic 必须与 actor 参数隔离才有"部署不变"保证);(2) critic 塔
只在 learner 的 GPU 上跑,占训练 step 时间 <30%,5090 吞吐富余(M3 实测 GPU
占空 90% 是推理服务,learner 更新只占 ~10% 墙钟);(3) A 允许 critic 塔独立
预训练(即本 gate 实验的产物可直接热启)。

### 1.4 训练侧改动清单

| 改动点 | 文件 | 内容 |
|---|---|---|
| 隐藏状态暴露 | `src/env_fixed.py` / baseline env | env 已持有 `self.hands[4]`(各家真实手牌)与 `self.tileWall`(墙列表);加一个只读方法 `oracle_state()` 返回 (hands, wall_counts) |
| 轨迹携带 oracle 通道 | `src/rl/env_worker.py` | 每决策点在现有 obs 之外调 `oracle_state()` 拼 28 通道,随轨迹进共享内存 arena(`TRAJ_SLOT_BYTES` 需扩 ~27%:obs 102→130 通道按 uint8 计,512KB 槽位仍够,需重核对 `write_traj` 布局)|
| learner | `src/rl/train_rl.py` | value 前向改走 critic 塔(130 入);PPO 损失、GAE、clip 全不变;优势仍由 critic 输出计算 |
| 推理服务 | `src/rl/infer_server.py` | rollout 时 worker 需要 value 做 GAE → infer 侧 'cur' 需挂 critic 塔;对手/eval 模型不需要 |
| 向听计算成本 | — | ×4 家/决策点,M1 已证缓存后吞吐占比 <30% 才触发查表化;先实测,必要时启用 `shanten_table`(已验证等价,opt-in)|

### 1.5 可行性 gate(本周 CPU 可执行)≪ 正在跑 ≫

`scripts/oracle_gate.py`:data.txt 重放 6000 局(~36 万决策点,oracle 通道从
牌谱完整重建),同一小型 CNN(W48/d4,~46 万参数)、同一预算(3 epoch),
基线 102ch vs oracle 130ch 各训一个 value 头,按局切分 held-out 报 Pearson。
**gate:oracle Pearson > 0.55**(M2 大网不完美信息基线 ~0.30 参照)。
通过 → GPU 期直接放大(全量 588 万样本 + W128/d10 critic 塔预训练);
不过 → 重新审视 oracle 特征(最可能缺:进张质量/牌墙时序信息)。

### 1.6 风险与缓解:oracle critic 的优势分布偏移

问题:oracle critic 给出的 V(s_oracle) 与 policy 实际可知信息不匹配,
GAE 优势 A = r + γV' − V 在"policy 看不出差别、oracle 看得出"的状态对上
产生系统性梯度——可能教 policy 学到"赌运气"行为(优势被 oracle 信息解释,
而非动作质量)。

缓解(按引入顺序):
1. **混合 baseline**:A_mix = α·A_oracle + (1−α)·A_imperfect,α 从 1.0 线性退火
   到 0.7;保留一个 102ch value 头同时训练(成本低),既做混合也做对照监控;
2. **oracle 信息 dropout 退火**:训练时以概率 p 把 oracle 28 通道整组置零
   (p: 0→0.3 退火),迫使 critic 不过度依赖全知,同时 dropout=1 分支即免费的
   不完美 critic;
3. **训练后期蒸馏回不完美 critic**(若需导出 value 供部署期搜索用——目前
   部署不用 value,此项最低优先)。
监控指标:两个 value 头的 explained variance 差值、A_oracle 与 A_imperfect
的相关系数(<0.5 报警)、对局层面赌博行为代理(点炮率突升)。

---

## 二、风格化对手池

### 2.1 动机(M3 实测证据)

eval_best 12h 无刷新触发早停,同期 vs RuleBot 仍创 +11.4 新高:对"自己过去
的风格"已收敛,对异质风格仍有提升空间。池 = 历史自我 + SL 锚点,风格单一。

### 2.2 三个偏科陪练(从 SL 冠军短训,数小时级)

奖励统一写成 `r = score/30 + β·shaping`,PPO 配置同 M3 试点(lr 4e-5,
KL 锚定 0.05 防崩坏),各训 4~6h(~500 updates,单风格单 GPU 夜间串行一晚):

| 风格 | shaping 公式 | β | 预期行为 |
|---|---|---|---|
| 激进型 aggro | `+1{自家胡} · (fan/8)`(胡牌按番数二次加权)| 0.5 | 抢胡、做大番,牺牲防守 |
| 防守型 turtle | `−1{点炮} · 2.0`(点炮在 score 之外重罚)| 0.5 | 弃牌保守、贴牌,胡率低点炮极低 |
| 碰碰胡型 pon | `+0.02·每次碰/明杠 + 1{胡且番含碰碰胡}·1.0` | 1.0 | 高频鸣牌、刻子流 |

验收:每个陪练 vs SL 锚点配对差值允许为负(偏科不求强),但行为指标必须
偏移(aggro 平均番 ≥+2、turtle 点炮率 ≤一半、pon 碰频 ≥2×),否则调 β 重训。

### 2.3 对手池注入机制(代码改动点)

| 改动 | 文件 | 内容 |
|---|---|---|
| 固定成员 | `src/rl/train_rl.py` | `cfg.pool_fixed = {'style_aggro': path, ...}`;resume/启动时 `load_model` 进 infer server,`ctrl['pool']` 拼接 `fixed + ['sl'] + pool_ids`;**淘汰逻辑只 pop 自我快照**(pool_ids),固定成员永不 drop(也就天然绕开淘汰竞态)|
| 采样权重 | `src/rl/env_worker.py` | `sample_assignment` 从均匀 `rng.choice(pool)` 改为带权:固定风格成员合计 ~25%、SL 10%、历史自我 65%(权重进 cfg)|
| 评测口径 | eval 协议 | 完整评测增加 vs 每个风格成员 500 组,监控"对某风格期望分"是否补齐 |

---

## 三、三周排程建议(若 6/20 晋级决赛)

| 周 | 内容 | GPU |
|---|---|---|
| W1(6/20-6/27)| oracle critic 全量实现+单元对拍;风格陪练 3 个短训+验收 | **有卡开机 ~3 天**(陪练短训一晚 + critic 塔预训练半天)|
| W2(6/27-7/4)| Gen2 主训练:CTDE PPO + 风格池,48~72h 跑两段,中段按 6h 摘要调参;并行做 fp16_resident 彩排与终选脚本复用 | **有卡连续 ~4 天** |
| W3(7/4-7/7)| 终选(final_select 流程换 Gen2 候选)+ 部署冻结(7/6 晚);留 1 天缓冲处理意外 | 无卡(评测 CPU)|

GPU 开机窗口合计 ~7 天;W1 前半周可先无卡做代码与对拍,有卡窗口集中后半周起。

---
*本文档结论标注纪律:gate 实验数字以 `logs/oracle_gate.log` 实测为准,写回本节。*
