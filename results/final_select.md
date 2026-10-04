# D-day 终选报告 (2026-06-13 06:49:50)

## 决策规则
```
预注册决策规则(执行前已定,不看结果改规则):
- 配对差值 >= +2*SE  -> 上 RL(显著更强)
- 落在 +-2*SE 内     -> 上 RL(同强度取训练更充分者,且 RL 对 RuleBot 更强)
- 配对差值 <= -2*SE  -> 上 SL-B,并告警深查 RL 训练
```

## a) 候选快筛(各 500 组 vs sl_v1_champion)
- eval_best.pt: 0.714 ± 0.648
- last.pt:      1.338 ± 0.637
- rl_final = **checkpoints/rl_v1/last.pt**

## b) 终选对决(2000 组复式,argmax 双方)
# Arena: nn:checkpoints/rl_v1/last.pt (x1) vs nn:checkpoints/sl_B_gn/best.pt — 1v3

- groups: 2000, candidate games: 8000, seed: 60614, workers: 24, elapsed: 1192.9s

| 指标 | 值 |
|---|---|
| 每局平均得分 | 1.055 ± 0.314 (SE) |
| 胡牌率 | 24.99% |
| 自摸率 | 7.21% |
| 点炮率 | 16.20% |
| 荒庄率 | 7.14% |
| 胡牌平均番数 | 13.47 |
| 同墙配对差值 | 1.407 ± 0.322 (SE, n=2000 组) |


## c) 锚点核对(1000 组 vs RuleBot)
# Arena: nn:checkpoints/rl_v1/last.pt (x1) vs rule — 1v3

- groups: 1000, candidate games: 4000, seed: 60615, workers: 24, elapsed: 228.8s

| 指标 | 值 |
|---|---|
| 每局平均得分 | 7.384 ± 0.490 (SE) |
| 胡牌率 | 33.50% |
| 自摸率 | 9.62% |
| 点炮率 | 10.85% |
| 荒庄率 | 12.20% |
| 胡牌平均番数 | 14.15 |
| 同墙配对差值 | 9.845 ± 0.610 (SE, n=1000 组) |

- 参照:sl_v1_champion vs RuleBot 为 +6.81±0.80(M2)

## 决策
- 终选配对差值 d = 1.407 ± 0.322(2SE 阈 = 0.644)
- **RL 上线(显著强于 SL,d >= +2SE)**
- 上线 ckpt:`checkpoints/rl_v1/last.pt`
## e) bundle 冒烟(30 局彩排 + 直接 spawn 内存)
```
{"response": "PASS", "debug": "{\"turn\": 1, \"turn_wall_ms\": 0.0, \"maxrss_kb\": 186388, \"model_tag\": \"rl_v1_B.fp16.pt|fp16_resident\", \"py\": \"3.6.13\", \"torch\": \"1.10.2+cpu\", \"torch_import_s\": 0.837, \"mahjonggb\": \"ok\", \"load_s\": 0.025, \"rss_after_load\": 186388}"}
 "games": 30,
 "invalid": 0,
 "crash": 0,
 "consistency": "1474/1474",
  "p50": 0.7,
  "p99": 87.39000000000006
```

## 待上传文件
- 用户存储空间:`dist/rl_v1_B.fp16.pt`(若决策为 SL-B 则无需上传,维持现版)
- bot 版本"+"更新:`dist/botzone_bot_rl_v1_B.zip`
    -rw-r--r-- 1 root root    19648 Jun 13 06:49 dist/botzone_bot_rl_v1_B.zip
    -rw-r--r-- 1 root root 28678642 Jun 13 06:49 dist/rl_v1_B.fp16.pt

## 加赛:last vs eval_best 直接对决(1000 组,仅供参考)
- 同墙配对差值:last 对 eval_best **+0.819±0.403**(>2SE)——更多训练胜过峰值快照,与快筛结论一致

## 待上传文件校验
- rl_v1_B.fp16.pt: 28678642 字节, md5 c9a16745928a27ddb808eedf97124547
- botzone_bot_rl_v1_B.zip: 19648 字节, md5 f6795191351cf7e3b049c086ffa4f13f
