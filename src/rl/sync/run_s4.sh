#!/bin/bash
# Step 4: 放大密集信号两 arm(都从 M5 起,β_kl=0.5,rounds 放大到 ~385≈3.5×Step3)。
#   arm pot_long — reward_mode=potential(Step 3 最佳配方,只加长)
#   arm combo    — reward_mode=combo(fanback 再分配 ⊕ potential 塑形,同长)
# 每 arm:长训(22w/480 局/轮,4 中间快照=5 评测点)-> 5 点各 arena@300 轨迹
#         -> 取轨迹峰值快照跑 arena@500(promote 峰值,非默认 last);峰值≠last 时 last 也评@500 作对比。
# 用法: bash src/rl/sync/run_s4.sh <ROUNDS>
set -u
cd /root/autodl-tmp/mahjong
ROUNDS=${1:?need ROUNDS}
M5=/root/autodl-tmp/mahjong/checkpoints/sl_v2_M5_gelu/best.pt
mkdir -p logs results/arena_d
COMMON="--workers 22 --games-per-round 480 --rounds $ROUNDS --snapshots 4 --set lr=1e-04 minibatch=4096 ppo_epochs=2 beta_kl=0.5"

get_diff() { grep 同墙 "$1" 2>/dev/null | awk -F'|' '{print $3}' | awk '{print $1}'; }

run_arm() {
  arm=$1; mode=$2
  dir=/root/autodl-tmp/mahjong/checkpoints/rl_s4_$arm
  echo "===== ARM $arm reward_mode=$mode rounds=$ROUNDS START $(date '+%F %T') ====="
  python src/rl/sync/train_rl.py $COMMON --out-dir "$dir" --reward-mode "$mode" \
     > logs/rl_s4_$arm.train.log 2>&1
  echo "[arm $arm] train exit=$?  -> arena @300 on 5 points $(date '+%T')"
  : > results/arena_d/rl_s4_${arm}_traj.txt
  best_ck=""; best_nm=""; best_diff=-999
  for ck in "$dir"/snap_*.pt "$dir"/last.pt; do
    [ -f "$ck" ] || continue
    nm=$(basename "$ck" .pt)
    md=results/arena_d/rl_s4_${arm}_${nm}_vs_M5.md
    python src/arena/run_arena.py --candidate nnv2:"$ck" --opponent nnv2:"$M5" \
       --groups 300 --workers 24 --seed 80640 --out "$md" > /dev/null 2>> logs/rl_s4_$arm.arena.log
    echo "$arm $nm | $(grep 同墙 "$md")" | tee -a results/arena_d/rl_s4_${arm}_traj.txt
    d=$(get_diff "$md")
    if [ -n "$d" ] && awk "BEGIN{exit !($d > $best_diff)}"; then best_diff=$d; best_ck="$ck"; best_nm=$nm; fi
  done
  echo "[arm $arm] PEAK=$best_nm diff=$best_diff -> arena @500 $(date '+%T')"
  python src/arena/run_arena.py --candidate nnv2:"$best_ck" --opponent nnv2:"$M5" \
     --groups 500 --workers 24 --seed 80640 \
     --out results/arena_d/rl_s4_${arm}_best_vs_M5.md > /dev/null 2>> logs/rl_s4_$arm.arena.log
  if [ "$best_nm" != "last" ]; then
    echo "[arm $arm] peak($best_nm)!=last -> 也评 last@500 作先升后崩对比"
    python src/arena/run_arena.py --candidate nnv2:"$dir"/last.pt --opponent nnv2:"$M5" \
       --groups 500 --workers 24 --seed 80640 \
       --out results/arena_d/rl_s4_${arm}_last_vs_M5.md > /dev/null 2>> logs/rl_s4_$arm.arena.log
  fi
  echo "===== ARM $arm DONE peak=$best_nm($best_diff) $(date '+%F %T') ====="
}

run_arm pot_long potential
run_arm combo combo
echo "ALL_ARMS_DONE $(date '+%F %T')"
