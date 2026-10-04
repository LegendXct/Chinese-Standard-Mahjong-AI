#!/usr/bin/env bash
# M1 服务器全量运行:预处理 -> 重放校验 -> pytest(增广200局)-> loader 吞吐 -> arena 冒烟
# 在 tmux 中执行:tmux new -d -s m1 'bash scripts/run_m1_server.sh'
set -uo pipefail
cd /root/autodl-tmp/mahjong
mkdir -p logs results tmp
{
echo "=== [1/6] full preprocess (128 workers) ==="
time python src/preprocess_v1.py --data baseline/SL/data/data.txt \
    --out /root/autodl-tmp/mahjong/data_v1 --workers 128
echo
echo "=== [2/6] full replay legality check (128 workers) ==="
time python src/replay_check.py --data baseline/SL/data/data.txt --workers 128 \
    --out results/replay_violations.json
echo "replay_check exit: $?"
echo
echo "=== [3/6] pytest full suite (AUG_GAMES=200) ==="
time AUG_GAMES=200 python -m pytest tests/ -q
echo
echo "=== [4/6] bench loader (workers 8/16/32) ==="
for w in 8 16 32; do
    python scripts/bench_loader.py --root /root/autodl-tmp/mahjong/data_v1 --workers $w
done
echo
echo "=== [5/6] arena: rule x1 vs random x3, 500 groups ==="
time python src/arena/run_arena.py --candidate rule --opponent random \
    --groups 500 --workers 64 --out results/arena_rule_vs_random.md
echo
echo "=== [6/6] arena: rule vs rule homogeneous, 500 groups ==="
time python src/arena/run_arena.py --candidate rule --opponent rule \
    --groups 500 --workers 64 --out results/arena_rule_vs_rule.md
echo
df -h /root/autodl-tmp | tail -1
du -sh /root/autodl-tmp/mahjong/data_v1
echo "=== ALL DONE ==="
} 2>&1 | tee logs/m1_run.log
