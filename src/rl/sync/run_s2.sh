#!/bin/bash
# Step 2: beta_kl 三 arm 小扫(A=0.02 / B=0.1 / C=0.5),其余超参完全一致。
# 每 arm:短训(22 worker,480 局/轮,2 中间快照)-> 快照+last 各跑 arena@200 轨迹 -> last 跑 arena@500。
# 三 arm 串行(训练用 22 核;arena 在该 arm 训练结束后串行跑,不与训练抢 CPU)。
# 用法: bash src/rl/sync/run_s2.sh <ROUNDS>
set -u
cd /root/autodl-tmp/mahjong
ROUNDS=${1:?need ROUNDS}
M5=/root/autodl-tmp/mahjong/checkpoints/sl_v2_M5_gelu/best.pt
mkdir -p logs results/arena_d
# 受控变量(三 arm 相同):lr 1e-4(实测 5e-5 太稳 clip~0.002 几乎不动,75min 难见信号;
#   1e-4 grad_norm 仍 ~0.3 << grad_clip 1.0,安全);minibatch 4096;epochs 2
COMMON="--workers 22 --games-per-round 480 --rounds $ROUNDS --snapshots 2 --set lr=1e-04 minibatch=4096 ppo_epochs=2"

run_arm () {
  arm=$1; beta=$2
  dir=/root/autodl-tmp/mahjong/checkpoints/rl_s2_beta$arm
  echo "===== ARM $arm beta=$beta rounds=$ROUNDS START $(date '+%F %T') ====="
  python src/rl/sync/train_rl.py $COMMON --out-dir "$dir" --beta-kl "$beta" \
     > logs/rl_s2_$arm.train.log 2>&1
  echo "[arm $arm] train exit=$?  -> arena snapshots+last @200 $(date '+%T')"
  : > results/arena_d/rl_s2_${arm}_traj.txt
  for ck in "$dir"/snap_*.pt "$dir"/last.pt; do
    [ -f "$ck" ] || continue
    nm=$(basename "$ck" .pt)
    python src/arena/run_arena.py --candidate nnv2:"$ck" --opponent nnv2:"$M5" \
       --groups 200 --workers 24 --seed 80640 \
       --out results/arena_d/rl_s2_${arm}_${nm}_vs_M5.md > /dev/null 2>> logs/rl_s2_$arm.arena.log
    line=$(grep '同墙配对差值' results/arena_d/rl_s2_${arm}_${nm}_vs_M5.md)
    echo "$arm $nm | $line" | tee -a results/arena_d/rl_s2_${arm}_traj.txt
  done
  echo "[arm $arm] arena full @500 on last $(date '+%T')"
  python src/arena/run_arena.py --candidate nnv2:"$dir"/last.pt --opponent nnv2:"$M5" \
     --groups 500 --workers 24 --seed 80640 \
     --out results/arena_d/rl_s2_${arm}_vs_M5.md > /dev/null 2>> logs/rl_s2_$arm.arena.log
  echo "===== ARM $arm DONE $(date '+%F %T') ====="
}

run_arm A 0.02
run_arm B 0.1
run_arm C 0.5
echo "ALL_ARMS_DONE $(date '+%F %T')"
