# M5(concat·GELU)部署彩排(botzone-compat,2026-06-14)

- 权重 = `checkpoints/sl_v2_M5_gelu/best.pt`(concat·GELU·GN·vec·去噪,val 0.89548)→ `dist/sl_v2_m5_gelu.fp16.pt`
- bot = `bot_main_v2`(fp16_resident;`FP16ResidentNetV2` 已加 GELU 支持);100 局协议级彩排。

| 门槛 | **M5 (sl_v2_m5_gelu)** | best-v2(M3′,参照) | 判定 |
|---|---|---|---|
| 常驻内存(直 spawn 权威) | **182.6 MB 加载 / 198.6 MB 推理峰** | 182.5 / 196.8 | ✅ <256,余 ~57 MB |
| 延迟 p99 / p95 / p50 | **86.8 / 69.5 / 0.9 ms** | 86.0 / 68.3 / 0.7 | ✅ <300 |
| 首回合 | **0.81 s** | 0.50 s | ✅ <1.5 |
| fp16_resident(GELU) vs fp32 一致性 | **5006/5006 = 100%** | 5154/5154 | ✅ ≥99.9% |
| 0 非法 / 0 崩溃(100 局) | **0 / 0** | 0 / 0 | ✅ |

- M5 与 best-v2 **同 concat 架构、同 14.5M、内存同档**,唯一差别 ReLU→GELU。`FP16ResidentNetV2` 加 GELU 后
  fp16↔fp32 **逐位一致(5006/5006)**,GELU 部署路径验证无误。
- **bundle md5**:`sl_v2_m5_gelu.fp16.pt` = `6e641e55652341affbc8a94e5c987c40`(29,007,314 B);
  `botzone_bot_sl_v2_m5_gelu.zip` = `957880bb79ceb42585bfd70af387c50f`(26,383 B)。
- **棋力(results/arena_d/)**:val 0.89548、arena vs rl_v1_B **+7.329±0.729**、vs best-v2(M3′)**+4.421±0.753**。
- 原始数据:`results/rehearsal_m5.json`。独立观察槽位,与 Round 1 的 rl_v1_B 完全隔离。
