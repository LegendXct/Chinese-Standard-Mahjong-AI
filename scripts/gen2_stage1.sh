#!/usr/bin/env bash
# gen2 阶段一:SL 基座消融(M0→M1→M2→M3)+ arena 对照。连续执行,中途不停。
#   tmux new -d -s gen2 'bash scripts/gen2_stage1.sh 2>&1 | tee logs/gen2_stage1.log'
# 隔离:全部 v2 新文件 + 新 checkpoint 目录,绝不碰 v1/rl_v1_B 部署物。
set -u
export OMP_NUM_THREADS=8   # 清掉 shell 残留的空 OMP_NUM_THREADS(libgomp 报 invalid)
cd /root/autodl-tmp/mahjong
DATA=/root/autodl-tmp/mahjong/data_v1
CKPT=/root/autodl-tmp/mahjong/checkpoints
SL=$CKPT/sl_B_gn/best.pt
RL=$CKPT/rl_v1/last.pt
W=24
mkdir -p logs results

tmux new -d -s sentinel "bash scripts/mem_sentinel.sh" 2>/dev/null || true

echo "===[1/8] find_asym_fans $(date +%T)==="
python scripts/find_asym_fans.py --data-txt baseline/SL/data/data.txt --out "$DATA"

echo "===[2/8] build_vec_v2 $(date +%T)==="
python src/build_vec_v2.py --data "$DATA" --workers "$W"

echo "===[3/8] M0 baseline (v1, B, GN) $(date +%T)==="
python src/train_sl.py --config B --norm gn --epochs 20 --workers "$W" \
    --data "$DATA" --run-dir "$CKPT/sl_v2_M0_gn"

echo "===[4/8] M1 (v1, B, BN) $(date +%T)==="
python src/train_sl.py --config B --norm bn --epochs 20 --workers "$W" \
    --data "$DATA" --run-dir "$CKPT/sl_v2_M1_bn"

echo "===[5/8] M2 (v2, B, BN, +vec) $(date +%T)==="
python src/train_sl_v2.py --config B --norm bn --epochs 20 --workers "$W" \
    --data "$DATA" --run-dir "$CKPT/sl_v2_M2_vec"

echo "===[6/8] M3 (v2, B, BN, +vec, +denoise) $(date +%T)==="
python src/train_sl_v2.py --config B --norm bn --epochs 20 --workers "$W" \
    --data "$DATA" --run-dir "$CKPT/sl_v2_M3_denoise" --exclude-asym

echo "===[7/8] pick best v2 ckpt $(date +%T)==="
BEST=$(python - <<'PY'
import torch, os
ck='/root/autodl-tmp/mahjong/checkpoints'
cands=[]
for d in ('sl_v2_M2_vec','sl_v2_M3_denoise'):
    p=os.path.join(ck,d,'best.pt')
    if os.path.exists(p):
        a=torch.load(p,map_location='cpu',weights_only=False).get('val_acc',-1)
        cands.append((a,p))
cands.sort()
print(cands[-1][1] if cands else '')
PY
)
echo "best v2 ckpt = $BEST"

echo "===[8/8] arena: best SL-v2 vs sl_champion & vs rl_v1_B (500组 each) $(date +%T)==="
python src/arena/run_arena.py --candidate "nnv2:$BEST" --opponent "nn:$SL" \
    --groups 500 --workers "$W" --seed 80613 --out results/sl_v2_vs_sl.md
python src/arena/run_arena.py --candidate "nnv2:$BEST" --opponent "nn:$RL" \
    --groups 500 --workers "$W" --seed 80614 --out results/sl_v2_vs_rl.md

echo "GEN2_STAGE1_DONE $(date +%T) best=$BEST"
