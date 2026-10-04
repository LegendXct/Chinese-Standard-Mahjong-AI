#!/usr/bin/env bash
# 阶段一-D 配对 arena:四配置(M3'/M5/M6/M7)统一同墙对比,arena 为最终裁决。
#   - 所有 "vs rl_v1_B" 用同一 seed(同墙)→ M3'/M5/M6/M7 配对可比(关键问题:谁 > best-v2 +2.25);
#   - 所有 "vs M3'" 用同一 seed;M3' vs M3' 恒 0 省略。
# 启动:tmux new -d -s gen2darena 'bash scripts/gen2d_arena.sh 2>&1 | tee logs/gen2d_arena.log'
set -u
export OMP_NUM_THREADS=8
cd /root/autodl-tmp/mahjong
CKPT=/root/autodl-tmp/mahjong/checkpoints
M3=$CKPT/sl_v2_M3_gn_denoise/best.pt
RL=$CKPT/rl_v1/last.pt
M5="nnv2:$CKPT/sl_v2_M5_gelu/best.pt"
M6="nnv2dual:$CKPT/sl_v2_M6_dual/best.pt"
M7="nnv2dual:$CKPT/sl_v2_M7_dual_gelu/best.pt"
W=24
SRL=80641   # 所有 vs rl_v1_B 同墙
SM3=80640   # 所有 vs M3' 同墙
mkdir -p results/arena_d
arena() { python src/arena/run_arena.py --candidate "$1" --opponent "$2" \
    --groups 500 --workers "$W" --seed "$3" --out "$4"; }

echo "===[A0] val 汇总(基准 M3'=0.83324)$(date +%T)==="
python - <<'PY'
import csv, os
ck = '/root/autodl-tmp/mahjong/checkpoints'
def best(d):
    p = os.path.join(ck, d, 'epoch_log.csv')
    if not os.path.exists(p):
        return None
    r = max(csv.DictReader(open(p)), key=lambda x: float(x['val_acc']))
    return float(r['val_acc']), float(r['val_play_acc'])
for n, d in (('M3p', 'sl_v2_M3_gn_denoise'), ('M5', 'sl_v2_M5_gelu'),
             ('M6', 'sl_v2_M6_dual'), ('M7', 'sl_v2_M7_dual_gelu')):
    r = best(d)
    if r:
        print('%-4s val=%.5f play=%.5f delta=%+.5f' % (n, r[0], r[1], r[0] - 0.83324))
PY

echo "===[A1] vs rl_v1_B(同墙 seed=$SRL,M3'/M5/M6/M7 配对可比)$(date +%T)==="
arena "nnv2:$M3" "nn:$RL" $SRL results/arena_d/M3p_vs_rl.md
arena "$M5"      "nn:$RL" $SRL results/arena_d/M5_vs_rl.md
arena "$M6"      "nn:$RL" $SRL results/arena_d/M6_vs_rl.md
arena "$M7"      "nn:$RL" $SRL results/arena_d/M7_vs_rl.md

echo "===[A2] vs M3'(同墙 seed=$SM3)$(date +%T)==="
arena "$M5" "nnv2:$M3" $SM3 results/arena_d/M5_vs_m3.md
arena "$M6" "nnv2:$M3" $SM3 results/arena_d/M6_vs_m3.md
arena "$M7" "nnv2:$M3" $SM3 results/arena_d/M7_vs_m3.md

echo "===[A3] 配对差一览(arena 为准;best-v2/M3' 历史 vs rl_v1_B = +2.250±0.687)$(date +%T)==="
echo "-- vs rl_v1_B --"
grep -H "同墙配对差值" results/arena_d/M3p_vs_rl.md results/arena_d/M5_vs_rl.md \
    results/arena_d/M6_vs_rl.md results/arena_d/M7_vs_rl.md
echo "-- vs M3' --"
grep -H "同墙配对差值" results/arena_d/M5_vs_m3.md results/arena_d/M6_vs_m3.md \
    results/arena_d/M7_vs_m3.md
echo "GEN2_ARENA_D_DONE $(date +%T)"
