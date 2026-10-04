#!/usr/bin/env bash
# 路径乙-2:vs-M5 仍恶化(≤−0.2 或继续下滑)时的"稳健优先"续训。
# 一次到位别小步磨:更强 KL 锚 M5(kl_sl_coef 0.10->0.15)+ 降 α(0.6->0.4) 先保对强者基本盘,
#   确认 vs-M5 回正后再谈做大牌。pow3 / cap24 / β0.15 / 地板 0.5 不变。
# env: INIT(默认 rl_v2c eval-best;可覆盖为 rl_v2b 的 +0.414 archive) RUNDIR(默认 rl_v2d)。
# 启动      :tmux new -d -s g2main "bash scripts/gen2_main_esc3.sh 2>&1 | tee -a logs/gen2_maind.log"
# 指定 init :tmux new -d -s g2main "INIT=/path/to.pt bash scripts/gen2_main_esc3.sh 2>&1 | tee -a logs/gen2_maind.log"
set -u
cd /root/autodl-tmp/mahjong
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
INIT=${INIT:-/root/autodl-tmp/mahjong/checkpoints/rl_v2c/eval_best.pt}
RUNDIR=${RUNDIR:-/root/autodl-tmp/mahjong/checkpoints/rl_v2d}
mkdir -p logs "$RUNDIR"

# 内存哨兵(同 gen2_main)
(
  over=0
  while true; do
    cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0); gb=$((cur / 1073741824))
    echo "[mem-sentinel] $(date +%H:%M:%S) cgroup=${gb}GB" >> logs/g2_mem_sentinel.log
    if [ "$gb" -gt 85 ]; then over=$((over + 1));
      [ "$over" -ge 3 ] && echo "[mem-sentinel] !!! WARN >85GB x${over} !!!" >> logs/g2_mem_sentinel.log
    else over=0; fi
    sleep 60
  done
) &
trap "kill $! 2>/dev/null" EXIT

echo "===== GEN2 MAIN esc3 yi-2 start $(date '+%F %T') init=$INIT run=$RUNDIR ====="
python src/rl/train_rl.py --mode main --set \
  run_dir=$RUNDIR init_model=$INIT \
  reward_alpha=0.4 reward_fan_pow=3.0 reward_fan_cap=24 reward_beta=0.15 \
  kl_sl_coef=0.15 eval_floor_se=0.5 \
  n_env_workers=22 envs_per_worker=16 max_hours=48
echo "===== GEN2 MAIN esc3 segment done $(date '+%F %T') ====="
