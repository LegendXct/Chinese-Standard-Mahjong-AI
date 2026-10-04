#!/usr/bin/env bash
# gen2 阶段一-C:扩展 vec 特征(feature_v2_ext,+10 维番种/七对/牌型先验)训 M4=GN+vec_ext,
# 与 M3'(83.3%) 比 val 增量,并 arena vs best-v2(真实棋力,准确率涨≠棋力涨)。
# 启动:tmux new -d -s gen2c 'bash scripts/gen2_stage1c.sh 2>&1 | tee logs/gen2_stage1c.log'
# 隔离:全部 v2.1 新文件 + 新 checkpoint,绝不碰 v1/rl_v1_B/已完成的 v2 产物。
set -u
export OMP_NUM_THREADS=8
cd /root/autodl-tmp/mahjong
DATA=/root/autodl-tmp/mahjong/data_v1
CKPT=/root/autodl-tmp/mahjong/checkpoints
BEST=$CKPT/sl_v2_M3_gn_denoise/best.pt    # 现最优 v2(基准)
RL=$CKPT/rl_v1/last.pt
W=24
mkdir -p logs results
tmux new -d -s sentinel "bash scripts/mem_sentinel.sh" 2>/dev/null || true

echo "===[C0] 冒烟 $(date +%T)==="
python scripts/gen2_ext_smoke.py "$DATA" || { echo "SMOKE FAILED, abort"; exit 1; }

echo "===[C1] build vec_ext.dat $(date +%T)==="
python src/build_vec_v2_ext.py --data "$DATA" --workers "$W"

echo "===[C2] M4 = v2.1 · GN · +vec_ext(20ep)$(date +%T)==="
python src/train_sl_v2.py --config B --norm gn --epochs 20 --workers "$W" \
    --data "$DATA" --run-dir "$CKPT/sl_v2_M4_gn_vecext" --vec-ext

echo "===[C3] M4 vs M3'(0.83324) val 增量 $(date +%T)==="
M4=$(python - <<'PY'
import csv
p = '/root/autodl-tmp/mahjong/checkpoints/sl_v2_M4_gn_vecext/epoch_log.csv'
print('%.5f' % max(float(r['val_acc']) for r in csv.DictReader(open(p))))
PY
)
echo "M4(GN+vec_ext) best val=$M4 ; M3'=0.83324 ; 增量=$(python -c "print('%.5f'%($M4-0.83324))")"

echo "===[C4] arena: M4 vs best-v2(M3') 500 组 $(date +%T)==="
python src/arena/run_arena.py --candidate "nnv2ext:$CKPT/sl_v2_M4_gn_vecext/best.pt" \
    --opponent "nnv2:$BEST" --groups 500 --workers "$W" --seed 80615 \
    --out results/m4_vs_bestv2.md

echo "===[C5] arena: M4 vs rl_v1_B 500 组(是否仍超成品)$(date +%T)==="
python src/arena/run_arena.py --candidate "nnv2ext:$CKPT/sl_v2_M4_gn_vecext/best.pt" \
    --opponent "nn:$RL" --groups 500 --workers "$W" --seed 80616 \
    --out results/m4_vs_rl.md

echo "GEN2_STAGE1C_DONE $(date +%T) M4=$M4"
