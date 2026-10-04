#!/usr/bin/env bash
# D-day 终选一键脚本:RL 收束后在服务器 tmux 中执行
#   tmux new -d -s final 'bash scripts/final_select.sh 2>&1 | tee logs/final_select.log'
# 前提:RL 主跑已结束(25 核全空);绝不在训练期间跑。
# 预计总耗时(24 worker):快筛 2x500 组 ~12min + 终选 2000 组 ~25min
#   + RuleBot 1000 组 ~5min + bundle/冒烟 30 局 ~4min ≈ 45~60 分钟。
set -euo pipefail
cd /root/autodl-tmp/mahjong

SL=checkpoints/sl_B_gn/best.pt
W=${WORKERS:-24}
TMP=tmp/final
OUT=results/final_select.md
COMPAT=/root/miniconda3/envs/botzone-compat/bin/python
mkdir -p "$TMP" results logs dist

# 注意:tmux server 的 cmdline 残留启动参数也会被 pgrep -f 命中,须按 comm 过滤真 python
for p in $(pgrep -f 'rl/train_rl.py' || true); do
    if [ "$(cat /proc/$p/comm 2>/dev/null)" = "python" ]; then
        echo 'FATAL: RL 主跑仍在运行,终选评测会抢 CPU,退出'; exit 1
    fi
done

echo "[1/5] $(date +%T) 候选快筛(eval_best/last 各 500 组 vs SL 冠军)"
python src/arena/run_arena.py --candidate nn:checkpoints/rl_v1/eval_best.pt \
    --opponent "nn:$SL" --groups 500 --workers "$W" --seed 60613 \
    --out "$TMP/screen_evalbest.md" > /dev/null
python src/arena/run_arena.py --candidate nn:checkpoints/rl_v1/last.pt \
    --opponent "nn:$SL" --groups 500 --workers "$W" --seed 60613 \
    --out "$TMP/screen_last.md" > /dev/null
RL_FINAL=$(python scripts/final_decide.py screen "$TMP")
echo "    rl_final = $RL_FINAL"

echo "[2/5] $(date +%T) 终选对决(2000 组 vs SL 冠军)"
python src/arena/run_arena.py --candidate "nn:$RL_FINAL" --opponent "nn:$SL" \
    --groups 2000 --workers "$W" --seed 60614 --out "$TMP/duel.md" > /dev/null

echo "[3/5] $(date +%T) 锚点核对(1000 组 vs RuleBot)"
python src/arena/run_arena.py --candidate "nn:$RL_FINAL" --opponent rule \
    --groups 1000 --workers "$W" --seed 60615 --out "$TMP/rule.md" > /dev/null

echo "[4/5] $(date +%T) 决策与报告 -> $OUT"
WINNER=$(python scripts/final_decide.py report "$RL_FINAL" "$SL" "$TMP" "$OUT")
echo "    winner = $WINNER"

echo "[5/5] $(date +%T) bundle + 冒烟(直接 spawn 内存实测 + 30 局彩排)"
python scripts/make_bundle.py --ckpt "$WINNER" --name rl_v1_B --infer-mode fp16_resident
echo '{"requests":["0 0 1"],"responses":[]}' \
    | BOT_WEIGHTS=dist/rl_v1_B.fp16.pt BOT_INFER_MODE=fp16_resident \
      "$COMPAT" src/deploy/bot_main.py | head -1 | tee "$TMP/rss_probe.json"
python src/deploy/rehearsal.py --weights dist/rl_v1_B.fp16.pt --games 30 \
    --tag rl_v1_B --infer-mode fp16_resident --bot-python "$COMPAT" \
    > "$TMP/smoke.json" 2> /dev/null
{
    echo '## e) bundle 冒烟(30 局彩排 + 直接 spawn 内存)'
    echo '```'
    cat "$TMP/rss_probe.json"
    grep -E '"(games|invalid|crash|consistency|p50|p99|first_turn)"' "$TMP/smoke.json" || cat "$TMP/smoke.json"
    echo '```'
    echo ''
    echo '## 待上传文件'
    echo '- 用户存储空间:`dist/rl_v1_B.fp16.pt`(若决策为 SL-B 则无需上传,维持现版)'
    echo '- bot 版本"+"更新:`dist/botzone_bot_rl_v1_B.zip`'
    ls -la dist/rl_v1_B.fp16.pt dist/botzone_bot_rl_v1_B.zip | sed 's/^/    /'
} >> "$OUT"
echo "FINAL_SELECT_DONE $(date +%T) winner=$WINNER 报告=$OUT"
