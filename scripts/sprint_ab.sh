#!/usr/bin/env bash
# Round 1 冲刺 A/B(无卡 0.5 核降规模版)。先跑修好的 oracle gate,再跑 3 臂 arena。
# 0.5 核实测 ~61s/组,故每臂仅 30 组(行为方向可看,分数 ±2SE 难达,如实标注)。
# 有卡开机后改用 scripts/final_select 同款 24 worker + 500 组拿显著结论。
set -u
cd /root/autodl-tmp/mahjong
RL=checkpoints/rl_v1/last.pt          # 已上线的保底 bot 权重
SL=checkpoints/sl_B_gn/best.pt        # 评测对手(冠军)
# 注意:GROUPS 是 bash 内建(用户组 id 数组),不能用作变量名,否则
# ${GROUPS:-30} 取到组 id 0 → 跑 0 组(已踩坑)。用 NG。
NG=${NG:-30}

if [ "${1:-}" != "--ab-only" ]; then
    echo "=== [0] oracle gate (fixed, 1-thread) ==="
    python scripts/oracle_gate.py --games 300 --val-games 40 --workers 1 --epochs 2 \
        2>&1 | tee logs/oracle_gate.log | grep -E "extracted|Pearson|gate"
fi

for arm in "argmax:nn:$RL" "T0.5:nn:$RL@t0.5" "claim:nn:$RL@claim"; do
    name=${arm%%:*}; spec=${arm#*:}
    echo "=== [$name] $spec vs sl_champion, $NG groups ==="
    python src/arena/run_arena.py --candidate "$spec" --opponent "nn:$SL" \
        --groups "$NG" --workers 1 --seed 70615 \
        --out "results/ab_$name.md" 2>&1 | grep -E "胡牌率|点炮率|平均番|配对差|groups"
done
echo "SPRINT_AB_DONE"
