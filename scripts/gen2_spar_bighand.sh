#!/usr/bin/env bash
# bighand 重训 — 非线性番奖励版:r += α·(min(fan,cap)/8)^pow,KL=0,纯自对弈。
# 用于 cycle-1 线性 α=0.8 验收不过后(线性对小牌也给奖励→理性选小牌,提 α 无效)。
#   pow=2 让高番奖励远大于低番(fan24 是 fan8 的 9 倍),cap=24 抑制大牌方差。
# env: BH_ALPHA(默认0.8) FAN_POW(默认2) FAN_CAP(默认24) UPD(默认300)
# 启动:tmux new -d -s g2bh 'bash scripts/gen2_spar_bighand.sh 2>&1 | tee logs/gen2_bighand_nl.log'
set -u
cd /root/autodl-tmp/mahjong
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
M5=/root/autodl-tmp/mahjong/checkpoints/sl_v2_M5_gelu/best.pt
SPAR=/root/autodl-tmp/mahjong/checkpoints/rl_v2_spar
mkdir -p "$SPAR" logs
BH_ALPHA=${BH_ALPHA:-0.8}; FAN_POW=${FAN_POW:-2.0}; FAN_CAP=${FAN_CAP:-24}; UPD=${UPD:-300}
echo "===== SPAR bighand NONLINEAR (alpha=$BH_ALPHA pow=$FAN_POW cap=$FAN_CAP KL=0 upd=$UPD) $(date +%T) ====="
rm -rf $SPAR/bighand_run
python src/rl/train_rl.py --mode pilot --set \
  arch=v2 sl_champion=$M5 p_pure_selfplay=1.0 w_bighand=0 w_def=0 w_rl_v1b=0 \
  rollout_size=16384 max_updates=$UPD max_hours=6 kl_sl_coef=0 \
  alarm_kl_sl=1e9 alarm_entropy_frac=0 alarm_hu_drop=1 \
  eval_every_min=1e9 eval_full_min=1e9 ckpt_every_updates=50 \
  run_dir=$SPAR/bighand_run reward_alpha=$BH_ALPHA reward_beta=0.15 \
  reward_fan_pow=$FAN_POW reward_fan_cap=$FAN_CAP
cp $SPAR/bighand_run/last.pt $SPAR/spar_bighand.pt && echo "frozen spar_bighand.pt (nonlinear pow=$FAN_POW)"
echo "===== GEN2_BIGHAND_NL_DONE $(date +%T) ====="
