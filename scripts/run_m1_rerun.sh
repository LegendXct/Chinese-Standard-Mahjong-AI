#!/usr/bin/env bash
# M1 重跑(shanten_py 替换库向听后受影响的环节)
set -uo pipefail
cd /root/autodl-tmp/mahjong
mkdir -p logs results tmp
{
echo "=== [1/4] full preprocess rerun (128 workers) ==="
rm -rf /root/autodl-tmp/mahjong/data_v1
time python src/preprocess_v1.py --data baseline/SL/data/data.txt \
    --out /root/autodl-tmp/mahjong/data_v1 --workers 128
echo
echo "=== [2/4] pytest full suite (AUG_GAMES=200) ==="
time AUG_GAMES=200 python -m pytest tests/ -q -s 2>&1 | grep -E "k=|passed|failed|FAILED|Error"
echo
echo "=== [3/4] bench loader (workers 16/32) ==="
for w in 16 32; do
    python scripts/bench_loader.py --root /root/autodl-tmp/mahjong/data_v1 --workers $w 2>/dev/null
done
echo
echo "=== [4/4] arena reruns, 500 groups each ==="
time python src/arena/run_arena.py --candidate rule --opponent random \
    --groups 500 --workers 64 --out results/arena_rule_vs_random.md
time python src/arena/run_arena.py --candidate rule --opponent rule \
    --groups 500 --workers 64 --out results/arena_rule_vs_rule.md
echo
df -h /root/autodl-tmp | tail -1
du -sh /root/autodl-tmp/mahjong/data_v1
echo "=== RERUN ALL DONE ==="
} 2>&1 | tee logs/m1_rerun.log
