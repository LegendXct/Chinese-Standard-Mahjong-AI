# gen2 计划(冲 7/7 决赛轮)

Round 1(6/15)维持已上线 **rl_v1_B 不动**(保底)。gen2 面向决赛轮,三阶段:

1. **阶段一:SL 基座提升 80.9%→~88%(本阶段)**
   诊断:同框架标杆 Arthals(纯 SL、20 层 ResNet、BN、15.58M)做到 ~89% 验证准确率 +
   天梯第一;我们 80.9% 差约 8pp。最大两个嫌疑:**GN(我们为 RL 稳定性选)vs BN
   (SL 更准)** 与 **缺"每弃牌质量"标量先验**。
   消融分步定位(每步 20ep,只改一项,便于归因):
   - M0 v1+B+GN(复现 80.9% 锚点)
   - M1 +BN(量化 BN 增量)
   - M2 +每弃牌质量标量 vec(model_v2 双输入)
   - M3 +增广去噪(绿一色/推不倒禁增广,对标 Arthals cannot_enhance_matches)
   产物全部隔离:feature_v2/model_v2/dataset_v2/build_vec_v2/train_sl_v2 + 新 checkpoint。
2. **阶段二:进攻导向 RL**(番数正比奖励 + 听牌未和惩罚 + 风格化对手池)
   依据我方天梯对局的聚合统计(胜率 11% vs 同桌均值 27%、平均番 10 vs 14)与第二份开源
   (invimirage:胜者 +fan、点炮者 −0.5fan 的不对称奖励)——奖励应**正比于番数**鼓励做大牌,
   点炮罚则取和牌奖励的分数(不对称,避免恐点炮)。详见 round1_sprint_finetune.md。
3. **阶段三:完美信息 critic(CTDE)**
   gate 预研:小规模 oracle value 头 Pearson 比基线 +46%(方向正),需全量重跑定阈。
   详见 gen2_oracle_critic.md。

## 关键架构决策(阶段一)
- **vec 是 obs 的确定性函数**(feature_v2.compute_vec_from_obs):训练建表/arena/部署
  单一真源,杜绝口径不一致;复用 v1 obs.dat 省 21.5GB 磁盘;arena 评 v2 无需改环境。
- vec = 34×3 每弃牌质量(弃后向听/进张未见/听牌)+ 少量 obs 可解码全局标量。
  其他家暗手牌数不在自视角 obs 中,本版不含(Arthals 实测此类全局边界收益小)。
- model_v2:v1 预激活 ResNet 主干 + vec 拼接 MLP 双头,norm 可配(SL 用 bn,RL 切 gn)。

## 隔离纪律
Round 1 保底 = rl_v1_B(dist/rl_v1_B.fp16.pt + 已上传),阶段一全程零改动其相关文件。
v2 若进决赛部署,需为 model_v2 写新的 fp16_resident 变体(届时再做)。
