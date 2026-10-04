#!/usr/bin/env bash
# 前置1 验收 arena(硬门槛):陪练 vs M5,同墙复式 400 组。
#   baseline = M5 自对弈(候选=M5,对手=M5)给出 M5 的胡率/平均番/点炮率基线;
#   bighand 验收:胜局平均番显著↑ 且 点炮率不暴涨;
#   def 验收:点炮率显著↓。
# 启动:tmux new -d -s g2spararena 'bash scripts/gen2_spar_arena.sh 2>&1 | tee logs/gen2_spar_arena.log'
set -u
cd /root/autodl-tmp/mahjong
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=8
M5=/root/autodl-tmp/mahjong/checkpoints/sl_v2_M5_gelu/best.pt
SPAR=/root/autodl-tmp/mahjong/checkpoints/rl_v2_spar
mkdir -p results
A(){ python src/arena/run_arena.py --candidate "$1" --opponent "$2" \
       --groups 400 --workers 24 --seed 90011 --out "$3"; }

echo "=== baseline: M5 self-play $(date +%T) ==="
A "nnv2:$M5" "nnv2:$M5" results/spar_baseline.md
echo "=== bighand vs M5 $(date +%T) ==="
A "nnv2:$SPAR/spar_bighand.pt" "nnv2:$M5" results/spar_bighand_vs_m5.md
echo "=== def vs M5 $(date +%T) ==="
A "nnv2:$SPAR/spar_def.pt" "nnv2:$M5" results/spar_def_vs_m5.md

echo "=== 验收汇总(候选自身指标;对照 baseline)$(date +%T) ==="
grep -H -E "胡牌率|胡牌平均番|点炮率" \
  results/spar_baseline.md results/spar_bighand_vs_m5.md results/spar_def_vs_m5.md
echo "=== GEN2_SPAR_ARENA_DONE $(date +%T) ==="
