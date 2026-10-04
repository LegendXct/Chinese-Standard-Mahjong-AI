#!/usr/bin/env bash
# gen2 阶段一-D:向 Arthals 对齐(双输入结构 + GELU),GN 基座,分步消融,arena 为准。
#   M5 = M3'+GELU(model_v2 + gelu)
#   M6 = M3'+dual(model_v2_dual,relu)
#   M7 = M6+GELU(model_v2_dual,gelu;最接近 Arthals 完整版,除我们用 GN)
#   每个接 arena vs best-v2(M3') + vs rl_v1_B(各 500 组);最终基座以 arena vs rl_v1_B 配对差为准。
# 启动:tmux new -d -s gen2d 'bash scripts/gen2_stage1d.sh 2>&1 | tee logs/gen2_stage1d.log'
# 隔离:全部新文件 + 新 checkpoint,绝不动 rl_v1_B / best-v2(M3') 既有产物。
set -u
export OMP_NUM_THREADS=8
cd /root/autodl-tmp/mahjong
DATA=/root/autodl-tmp/mahjong/data_v1
CKPT=/root/autodl-tmp/mahjong/checkpoints
M3=$CKPT/sl_v2_M3_gn_denoise/best.pt
RL=$CKPT/rl_v1/last.pt
W=24
mkdir -p logs results
tmux new -d -s sentinel "bash scripts/mem_sentinel.sh" 2>/dev/null || true

echo "===[D0] 冒烟 $(date +%T)==="
python scripts/gen2_dual_smoke.py "$DATA" || { echo "SMOKE FAILED, abort"; exit 1; }

echo "===[D1] M5 = M3'+GELU (v2·GN·gelu·去噪) $(date +%T)==="
python src/train_sl_v2.py --config B --norm gn --act gelu --exclude-asym --epochs 20 \
    --workers "$W" --data "$DATA" --run-dir "$CKPT/sl_v2_M5_gelu"

echo "===[D2] M6 = M3'+dual (dual·GN·relu·去噪) $(date +%T)==="
python src/train_sl_v2.py --config B --norm gn --act relu --dual --exclude-asym --epochs 20 \
    --workers "$W" --data "$DATA" --run-dir "$CKPT/sl_v2_M6_dual"

echo "===[D3] M7 = M6+GELU (dual·GN·gelu·去噪) $(date +%T)==="
python src/train_sl_v2.py --config B --norm gn --act gelu --dual --exclude-asym --epochs 20 \
    --workers "$W" --data "$DATA" --run-dir "$CKPT/sl_v2_M7_dual_gelu"

echo "===[D4] val 增量汇总(基准 M3'=0.83324)$(date +%T)==="
python - <<'PY'
import csv, os
ck = '/root/autodl-tmp/mahjong/checkpoints'
def best(d):
    p = os.path.join(ck, d, 'epoch_log.csv')
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    r = max(rows, key=lambda x: float(x['val_acc']))
    return float(r['val_acc']), float(r['val_play_acc'])
for name, d in (('M3p', 'sl_v2_M3_gn_denoise'), ('M5', 'sl_v2_M5_gelu'),
                ('M6', 'sl_v2_M6_dual'), ('M7', 'sl_v2_M7_dual_gelu')):
    r = best(d)
    if r:
        print('%-4s %-22s val=%.5f play=%.5f delta=%+.5f' % (
            name, d, r[0], r[1], r[0] - 0.83324))
PY

echo "===[D5] arena(各 500 组;M5 用 nnv2,M6/M7 用 nnv2dual)$(date +%T)==="
run() { python src/arena/run_arena.py --candidate "$1" --opponent "$2" \
    --groups 500 --workers "$W" --seed "$3" --out "$4"; }
run "nnv2:$CKPT/sl_v2_M5_gelu/best.pt"          "nnv2:$M3" 80631 results/m5_vs_m3.md
run "nnv2:$CKPT/sl_v2_M5_gelu/best.pt"          "nn:$RL"   80632 results/m5_vs_rl.md
run "nnv2dual:$CKPT/sl_v2_M6_dual/best.pt"      "nnv2:$M3" 80633 results/m6_vs_m3.md
run "nnv2dual:$CKPT/sl_v2_M6_dual/best.pt"      "nn:$RL"   80634 results/m6_vs_rl.md
run "nnv2dual:$CKPT/sl_v2_M7_dual_gelu/best.pt" "nnv2:$M3" 80635 results/m7_vs_m3.md
run "nnv2dual:$CKPT/sl_v2_M7_dual_gelu/best.pt" "nn:$RL"   80636 results/m7_vs_rl.md

echo "===[D6] 配对差一览(arena 为准,M3' vs rl_v1_B 基准 +2.250±0.687)$(date +%T)==="
grep -H "同墙配对差值" results/m5_vs_m3.md results/m5_vs_rl.md \
    results/m6_vs_m3.md results/m6_vs_rl.md results/m7_vs_m3.md results/m7_vs_rl.md

echo "GEN2_STAGE1D_DONE $(date +%T)"
