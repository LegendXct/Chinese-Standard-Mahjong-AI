#!/usr/bin/env bash
# 阶段二 v2+vec 集成校验门禁:
#   gate-smoke = 纯自对弈跑 N 局,报零和/胡率/吞吐 + actor 计时拆分(含新增 vec 桶)= 前置2 profiling
#   gate-zero  = 收满 1 rollout,lr=0 更新验证权重不变 + KL(π‖M5)≈0 + 真 lr 首更新 ratio≈1/clip 极小
# 启动:tmux new -d -s g2gate 'bash scripts/gen2_gates.sh 2>&1 | tee logs/gen2_gates.log'
set -u
cd /root/autodl-tmp/mahjong
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
mkdir -p logs checkpoints/rl_v2_spar
M5=/root/autodl-tmp/mahjong/checkpoints/sl_v2_M5_gelu/best.pt
GATE=/root/autodl-tmp/mahjong/checkpoints/rl_v2_gate
C="arch=v2 sl_champion=$M5 run_dir=$GATE p_pure_selfplay=1.0 w_bighand=0 w_def=0 w_rl_v1b=0"

echo "===== GATE-SMOKE (v2+vec 纯自对弈,前置2 吞吐 profiling) $(date +%T) ====="
python src/rl/train_rl.py --mode gate-smoke --smoke-games 300 --set $C

echo "===== GATE-ZERO (v2+vec PPO 正确性) $(date +%T) ====="
python src/rl/train_rl.py --mode gate-zero --set $C

echo "===== GEN2_GATES_DONE $(date +%T) ====="
