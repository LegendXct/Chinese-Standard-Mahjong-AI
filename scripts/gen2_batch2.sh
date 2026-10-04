#!/usr/bin/env bash
# 阶段二 第二批(v1 验证机制):番数敏感攻守双塑形,从 rl_v2c eval-best 续训。
#  进攻 fan backward:c_fb=0.3 / γ0.9 / min12 / cap24,全步反传不分类型,叠加 GAE(只动 adv);
#  关单步番奖励 α=0(取代,避免混淆);防守番数敏感点炮罚 p_def=2.0 / cap24;
#  稳健保护:KL锚M5 0.10 / eval-best地板−0.5SE / 多样对手池 / 点炮总率红线 base+3pp;
#  value warmup(前40 upd value_coef×2,fan backward 抬 adv 方差)。绝不动 rl_v1_B。
# 启动:tmux new -d -s g2b2 'bash scripts/gen2_batch2.sh 2>&1 | tee -a logs/gen2_batch2.log'
# 续训:tmux new -d -s g2b2 'bash scripts/gen2_batch2.sh resume 2>&1 | tee -a logs/gen2_batch2.log'
set -u
cd /root/autodl-tmp/mahjong
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
mkdir -p logs checkpoints/rl_v2d
RESUME=""
[ "${1:-}" = "resume" ] && RESUME="--resume"

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

echo "===== GEN2 BATCH2 fan-backward start $(date '+%F %T') RESUME=$RESUME ====="
python src/rl/train_rl.py --mode main $RESUME --set \
  run_dir=/root/autodl-tmp/mahjong/checkpoints/rl_v2d \
  init_model=/root/autodl-tmp/mahjong/checkpoints/rl_v2c_evalbest_0345.pt \
  reward_alpha=0.0 reward_beta=0.15 \
  dealin_pow=2.0 dealin_fan_cap=24 \
  fan_backward_coef=0.3 fan_backward_gamma=0.9 fan_backward_min_fan=12 fan_backward_fan_cap=24 \
  kl_sl_coef=0.10 eval_floor_se=0.5 \
  value_warmup_updates=40 value_warmup_mult=2.0 \
  n_env_workers=22 envs_per_worker=16 max_hours=48
echo "===== GEN2 BATCH2 segment done $(date '+%F %T') ====="
