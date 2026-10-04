#!/usr/bin/env bash
# 前置1:偏科陪练短训(M5 纯自对弈 + 番数塑形)。
# ★关键(cycle0 教训):kl_sl_coef=0 —— 陪练要主动漂移成偏科风格,KL 锚定 M5(0.05)
#   会把策略钉在 M5 旁(cycle0 KL 仅 0.032、arena 与 M5 无异)。故放开 KL,并关掉
#   会因"故意偏科"误触的 KL/熵/胡率告警。
# 参数(env 覆盖):BH_ALPHA(高番型 α,默认 0.8)、DEF_BETA(防守型 β,默认 0.6)、UPD(默认 300)。
# 启动:tmux new -d -s g2spar 'BH_ALPHA=0.8 bash scripts/gen2_spar.sh 2>&1 | tee logs/gen2_spar.log'
set -u
cd /root/autodl-tmp/mahjong
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
M5=/root/autodl-tmp/mahjong/checkpoints/sl_v2_M5_gelu/best.pt
SPAR=/root/autodl-tmp/mahjong/checkpoints/rl_v2_spar
mkdir -p "$SPAR" logs
BH_ALPHA=${BH_ALPHA:-0.8}
DEF_BETA=${DEF_BETA:-0.6}
UPD=${UPD:-300}
# 短训只求风格偏移;缩小 rollout 加速;KL=0 放开漂移;关误触告警
C="arch=v2 sl_champion=$M5 p_pure_selfplay=1.0 w_bighand=0 w_def=0 w_rl_v1b=0 \
   rollout_size=16384 max_updates=$UPD max_hours=6 kl_sl_coef=0 \
   alarm_kl_sl=1e9 alarm_entropy_frac=0 alarm_hu_drop=1 \
   eval_every_min=1e9 eval_full_min=1e9 ckpt_every_updates=50"

echo "===== SPAR bighand (alpha=$BH_ALPHA beta=0.15, KL=0, upd=$UPD) $(date +%T) ====="
rm -rf $SPAR/bighand_run
python src/rl/train_rl.py --mode pilot --set $C \
  run_dir=$SPAR/bighand_run reward_alpha=$BH_ALPHA reward_beta=0.15
cp $SPAR/bighand_run/last.pt $SPAR/spar_bighand.pt && echo "frozen spar_bighand.pt (alpha=$BH_ALPHA)"

echo "===== SPAR def (alpha=0.1 beta=$DEF_BETA, KL=0, upd=$UPD) $(date +%T) ====="
rm -rf $SPAR/def_run
python src/rl/train_rl.py --mode pilot --set $C \
  run_dir=$SPAR/def_run reward_alpha=0.1 reward_beta=$DEF_BETA
cp $SPAR/def_run/last.pt $SPAR/spar_def.pt && echo "frozen spar_def.pt (beta=$DEF_BETA)"

echo "===== GEN2_SPAR_DONE $(date +%T) ====="
