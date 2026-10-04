#!/bin/bash
# Step 3: 改奖励/信用分配两 arm 对照(唯一变量 = reward_mode),其余超参 = Step 2 紧锚(beta_kl=0.5)。
#   arm fanback   — 终局零和 R 沿轨迹反向折扣再分配(量级守恒,GAE 拿密集信号)
#   arm potential — 终局 R + 势能塑形 c·(gamma·Φ'−Φ),Φ=(6−shanten)/6,终局势能0(Ng et al. 保策略不变)
# 每 arm:短训(22w/480 局/轮,2 中间快照)-> 快照+last arena@200 轨迹 -> last arena@500。
# 用法: bash src/rl/sync/run_s3.sh <ROUNDS>
set -u
cd /root/autodl-tmp/mahjong
ROUNDS=${1:?need ROUNDS}
M5=/root/autodl-tmp/mahjong/checkpoints/sl_v2_M5_gelu/best.pt
mkdir -p logs results/arena_d
# 受控变量(两 arm 相同)= Step 2:lr 1e-4 / minibatch 4096 / epochs 2 / beta_kl 0.5(最稳紧锚)
COMMON="--workers 22 --games-per-round 480 --rounds $ROUNDS --snapshots 2 --set lr=1e-04 minibatch=4096 ppo_epochs=2 beta_kl=0.5"

run_arm () {
  arm=$1; mode=$2
  dir=/root/autodl-tmp/mahjong/checkpoints/rl_s3_$arm
  echo "===== ARM $arm reward_mode=$mode rounds=$ROUNDS START $(date '+%F %T') ====="
  python src/rl/sync/train_rl.py $COMMON --out-dir "$dir" --reward-mode "$mode" \
     > logs/rl_s3_$arm.train.log 2>&1
  echo "[arm $arm] train exit=$?  -> arena @200 snapshots+last $(date '+%T')"
  : > results/arena_d/rl_s3_${arm}_traj.txt
  for ck in "$dir"/snap_*.pt "$dir"/last.pt; do
    [ -f "$ck" ] || continue
    nm=$(basename "$ck" .pt)
    python src/arena/run_arena.py --candidate nnv2:"$ck" --opponent nnv2:"$M5" \
       --groups 200 --workers 24 --seed 80640 \
       --out results/arena_d/rl_s3_${arm}_${nm}_vs_M5.md > /dev/null 2>> logs/rl_s3_$arm.arena.log
    echo "$arm $nm | $(grep 同墙 results/arena_d/rl_s3_${arm}_${nm}_vs_M5.md)" \
       | tee -a results/arena_d/rl_s3_${arm}_traj.txt
  done
  echo "[arm $arm] arena full @500 on last $(date '+%T')"
  python src/arena/run_arena.py --candidate nnv2:"$dir"/last.pt --opponent nnv2:"$M5" \
     --groups 500 --workers 24 --seed 80640 \
     --out results/arena_d/rl_s3_${arm}_vs_M5.md > /dev/null 2>> logs/rl_s3_$arm.arena.log
  echo "===== ARM $arm DONE $(date '+%F %T') ====="
}

run_arm fanback fanback
run_arm potential potential
echo "ALL_ARMS_DONE $(date '+%F %T')"
