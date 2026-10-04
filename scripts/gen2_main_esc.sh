#!/usr/bin/env bash
# 路径乙:进攻加码续训(α=0.8 + 指数 pow=3,即 α·(fan/8)³;β=0.15 线性不变)。
# env: ALPHA(默认0.8) POW(默认3.0) INIT(init_model,默认 upd200) RUNDIR(默认 rl_v2c)
# 启动:tmux new -d -s g2main 'bash scripts/gen2_main_esc.sh 2>&1 | tee -a logs/gen2_mainc.log'
set -u
cd /root/autodl-tmp/mahjong
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
ALPHA=${ALPHA:-0.8}
POW=${POW:-3.0}
INIT=${INIT:-/root/autodl-tmp/mahjong/checkpoints/rl_v2_init.pt}
RUNDIR=${RUNDIR:-/root/autodl-tmp/mahjong/checkpoints/rl_v2c}
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

echo "===== GEN2 MAIN ESC start $(date '+%F %T') alpha=$ALPHA pow=$POW init=$INIT run=$RUNDIR ====="
python src/rl/train_rl.py --mode main --set \
  run_dir=$RUNDIR init_model=$INIT \
  reward_alpha=$ALPHA reward_fan_pow=$POW reward_fan_cap=24 reward_beta=0.15 \
  n_env_workers=22 envs_per_worker=16 max_hours=48
echo "===== GEN2 MAIN segment done $(date '+%F %T') ====="
