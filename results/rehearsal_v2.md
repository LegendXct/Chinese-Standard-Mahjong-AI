# best-v2 部署彩排(botzone-compat,2026-06-14)

- 权重 = `checkpoints/sl_v2_M3_gn_denoise/best.pt`(GN+vec,val 0.8332)→ `dist/sl_v2_best.fp16.pt`
- bot = `bot_main_v2`(fp16_resident,vec 由 compute_vec_from_obs 现算)
- 协议级彩排:`env_fixed` 判官,bot 坐 0 号位,其余 3 座 NNPolicyV2 argmax;100 局。

| 门槛 | **best-v2 (sl_v2_best)** | rl_v1_B(参照) | 判定 |
|---|---|---|---|
| 常驻内存(直 spawn 权威) | **182.5 MB 加载 / 196.8 MB 推理峰** | 186.7 MB | ✅ <256,余 ~59 MB |
| 延迟 p99 / p95 / p50 | **86.0 / 68.3 / 0.7 ms** | 85.3 ms (p99) | ✅ <300 |
| 首回合 | **0.50 s** | 0.719 s | ✅ <1.5 |
| fp16_resident vs fp32 一致性 | **5154/5154 (100%)** | 1000/1000 | ✅ ≥99.9% |
| 0 非法 / 0 崩溃(100 局) | **0 / 0** | 0 / 0 | ✅ |

- **内存权威口径** = shell 直接 spawn 的 `rss_after_load`;rehearsal 报的 491.7 MB 是 fork 自大驱动的
  COW 假象(沿用 v1 那条内存测量纪律)。
- **bundle md5**:`sl_v2_best.fp16.pt` = `ca0330512bbdd5b91cbb4492f4bb1487`(27.66 MB,29,006,821 B);
  `botzone_bot_sl_v2_best.zip` = `1db51257c32fd00077fa31f87471e72b`(26,267 B)。
- **独立 bot 槽位,与 Round 1 的 rl_v1_B 完全隔离。** 原始数据:`results/rehearsal_v2.json`。
- 注:首次 100 局彩排被 C 的 M4 训练(24 worker)饿死跑了 ~1h45 未完,清场后空载重跑 8.5 min 干净完成,
  非 bot 缺陷。
