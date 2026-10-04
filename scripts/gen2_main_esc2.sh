#!/usr/bin/env bash
# 路径乙(修正版):针对 vs-M5 恶化的进攻加码。从 rl_v2b 的 eval-best(upd156,+0.414)续训。
# 三改:进攻适度加码 α=0.6+pow3(奖励集中在真高番,避免搏不成乱打) /
#       加大 KL 锚 M5 kl_sl_coef=0.10(护稳健,防为搏大牌牺牲对强对手基本盘) /
#       eval-best 地板收紧 eval_floor_se=0.5(必须对每个强对手都不明显输,杜绝靠 def 撑均值)。
# run_dir=rl_v2c(独立,不动 rl_v2b 的 eval-best 保底)。
# 启动:tmux new -d -s g2main 'bash scripts/gen2_main_esc2.sh 2>&1 | tee -a logs/gen2_mainc.log'
set -u
cd /root/autodl-tmp/mahjong
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
mkdir -p logs checkpoints/rl_v2c

# 内存哨兵(同 gen2_main:每 60s 记 cgroup,连续 3 次 >85GB 告警,不杀任务)
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

echo "===== GEN2 MAIN esc2 corrected-yi start $(date '+%F %T') ====="
python src/rl/train_rl.py --mode main --set \
  run_dir=/root/autodl-tmp/mahjong/checkpoints/rl_v2c \
  init_model=/root/autodl-tmp/mahjong/checkpoints/rl_v2b/eval_best.pt \
  reward_alpha=0.6 reward_fan_pow=3.0 reward_fan_cap=24 reward_beta=0.15 \
  kl_sl_coef=0.10 eval_floor_se=0.5 \
  n_env_workers=22 envs_per_worker=16 max_hours=48
echo "===== GEN2 MAIN esc2 segment done $(date '+%F %T') ====="
