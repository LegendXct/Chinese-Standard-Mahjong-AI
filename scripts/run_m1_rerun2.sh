#!/usr/bin/env bash
# M1 重启流水线(内存修复版):预处理(--resume)→ 重放校验 → pytest → 吞吐 → arena
# 用法:WORKERS=96 tmux new -d -s m1 'bash scripts/run_m1_rerun2.sh'
set -uo pipefail
cd /root/autodl-tmp/mahjong
mkdir -p logs results tmp
WORKERS="${WORKERS:-96}"

# 内存哨兵:每 60s 记录 cgroup 用量,连续 3 次 >85GB 写告警(不杀任务,便于归因)
(
    over=0
    while true; do
        cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
        gb=$((cur / 1073741824))
        echo "[mem-sentinel] $(date +%H:%M:%S) cgroup=${gb}GB" >> logs/mem_sentinel.log
        if [ "$gb" -gt 85 ]; then
            over=$((over + 1))
            if [ "$over" -ge 3 ]; then
                echo "[mem-sentinel] !!!! WARNING: cgroup memory >85GB x${over} consecutive ($(date)) !!!!" | tee -a logs/mem_sentinel.log >> logs/m1_rerun2.log
            fi
        else
            over=0
        fi
        sleep 60
    done
) &
SENTINEL_PID=$!
trap "kill $SENTINEL_PID 2>/dev/null" EXIT

{
echo "=== [1/5] full preprocess --resume (workers=$WORKERS, chunks=256 与原跑对齐) ==="
time python src/preprocess_v1.py --data baseline/SL/data/data.txt \
    --out /root/autodl-tmp/mahjong/data_v1 --workers "$WORKERS" --chunks 256 --resume
echo
echo "=== [2/5] full replay legality check (workers=$WORKERS) ==="
time python src/replay_check.py --data baseline/SL/data/data.txt --workers "$WORKERS" \
    --out results/replay_violations.json
echo "replay_check exit: $?"
echo
echo "=== [3/5] pytest full suite (AUG_GAMES=200) ==="
time AUG_GAMES=200 python -m pytest tests/ -q -s 2>&1 | grep -E "k=|passed|failed|FAILED|Error"
echo
echo "=== [4/5] bench loader (workers 16/32) ==="
for w in 16 32; do
    python scripts/bench_loader.py --root /root/autodl-tmp/mahjong/data_v1 --workers $w 2>/dev/null
done
echo
echo "=== [5/5] arena: rule vs random + rule vs rule, 500 groups each ==="
time python src/arena/run_arena.py --candidate rule --opponent random \
    --groups 500 --workers 64 --out results/arena_rule_vs_random.md
time python src/arena/run_arena.py --candidate rule --opponent rule \
    --groups 500 --workers 64 --out results/arena_rule_vs_rule.md
echo
df -h /root/autodl-tmp | tail -1
du -sh /root/autodl-tmp/mahjong/data_v1
echo "=== RERUN2 ALL DONE ==="
} 2>&1 | tee logs/m1_rerun2.log
