#!/usr/bin/env bash
# 阶段二主训练 第一批:M5 热启动 + 番数正比奖励(α=0.3/β=0.15)+ 校准版对手池。
# config.py 默认已是校准值(arch=v2, sl=M5, run_dir=rl_v2, p=0.65,
#   w 自55/锚15/高番13/防12/rlv1b5, α0.3/β0.15, KL锚M5, lr2e-5)。
# 本脚本仅设 worker 规模(对齐 rl_v1 proven 22×16)+ max_hours,带内存哨兵。
# 启动 :tmux new -d -s g2main 'bash scripts/gen2_main.sh 2>&1 | tee -a logs/gen2_main.log'
# 续训 :tmux new -d -s g2main 'bash scripts/gen2_main.sh resume 2>&1 | tee -a logs/gen2_main.log'
set -u
cd /root/autodl-tmp/mahjong
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
mkdir -p logs checkpoints/rl_v2b
RESUME=""
[ "${1:-}" = "resume" ] && RESUME="--resume"

# 内存哨兵:每 60s 记录 cgroup 用量,连续 3 次 >85GB 告警(不杀任务,便于归因)
(
  over=0
  while true; do
    cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    gb=$((cur / 1073741824))
    echo "[mem-sentinel] $(date +%H:%M:%S) cgroup=${gb}GB" >> logs/g2_mem_sentinel.log
    if [ "$gb" -gt 85 ]; then
      over=$((over + 1))
      [ "$over" -ge 3 ] && echo "[mem-sentinel] !!! WARN cgroup >85GB x${over} ($(date)) !!!" \
        | tee -a logs/g2_mem_sentinel.log
    else over=0; fi
    sleep 60
  done
) &
trap "kill $! 2>/dev/null" EXIT

# 第一批-b:从 upd222 热启动 + (a)eval-best 竞争子集口径 + (b)非线性进攻奖励 α·(fan/8)²。
# run_dir=rl_v2b(新,不动 rl_v2 记录);init_model=upd222;KL 锚点仍 M5(sl_champion 默认)。
echo "===== GEN2 MAIN batch1b start $(date '+%F %T') RESUME=$RESUME ====="
python src/rl/train_rl.py --mode main $RESUME --set \
  run_dir=/root/autodl-tmp/mahjong/checkpoints/rl_v2b \
  init_model=/root/autodl-tmp/mahjong/checkpoints/rl_v2_init.pt \
  reward_alpha=0.5 reward_fan_pow=2.0 reward_fan_cap=24 reward_beta=0.15 \
  n_env_workers=22 envs_per_worker=16 max_hours=48
echo "===== GEN2 MAIN segment done $(date '+%F %T') ====="
