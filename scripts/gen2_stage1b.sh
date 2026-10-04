#!/usr/bin/env bash
# gen2 阶段一(修正链 v3):消融精细度匹配改动分量。
#   - vec(100+ 维实质特征):值得查 norm×vec 交互 => M2(BN+vec) 与 M2'(GN+vec) 都作基座跑,
#     二者之差即交互诊断(无需额外跑,两者本就都跑)。
#   - 去噪(仅 0.54% 样本,微改动):不值得查 norm×去噪 => M3 只跑 GN 版,不跑 M3-BN。
# 整条为无条件线性序列(无 fork 训练):等 M2 收尾 -> M2'(GN+vec) -> [诊断] -> M3'(GN+vec+去噪)
#   -> [去噪净增量] -> 从 {M0,M1,M2,M2',M3'} 选 val 最高 best-v2(自动辨 v1/v2)-> arena。
# 启动:tmux new -d -s gen2b 'bash scripts/gen2_stage1b.sh 2>&1 | tee logs/gen2_stage1b.log'
# 隔离:全部 v2 新文件 + 新 checkpoint 目录,绝不碰 v1/rl_v1_B 部署物。
set -u
export OMP_NUM_THREADS=8
cd /root/autodl-tmp/mahjong
DATA=/root/autodl-tmp/mahjong/data_v1
CKPT=/root/autodl-tmp/mahjong/checkpoints
SL=$CKPT/sl_B_gn/best.pt
RL=$CKPT/rl_v1/last.pt
W=24
mkdir -p logs results
tmux new -d -s sentinel "bash scripts/mem_sentinel.sh" 2>/dev/null || true

best_acc() {  # $1=run-dir 名;打印该目录 epoch_log.csv 的最大 val_acc(无则 -1)
    python - "$1" <<'PY'
import os, csv, sys
p = os.path.join('/root/autodl-tmp/mahjong/checkpoints', sys.argv[1], 'epoch_log.csv')
m = -1.0
if os.path.exists(p):
    with open(p) as f:
        for row in csv.DictReader(f):
            m = max(m, float(row['val_acc']))
print('%.5f' % m)
PY
}

echo "===[B0] 等 M2(BN+vec) 收尾 $(date +%T)==="
until grep -q 'TRAIN_SL_V2_DONE.*sl_v2_M2_vec' logs/gen2_stage1.log 2>/dev/null; do
    sleep 60
done
echo "===[B0] M2 已完成,开始 GN 链 $(date +%T)==="

echo "===[B1] M2' (v2, B, GN, +vec) $(date +%T)==="
python src/train_sl_v2.py --config B --norm gn --epochs 20 --workers "$W" \
    --data "$DATA" --run-dir "$CKPT/sl_v2_M2p_gn_vec"

echo "===[诊断] norm×vec 交互:M2(BN+vec) vs M2'(GN+vec) best val_acc $(date +%T)==="
M2=$(best_acc sl_v2_M2_vec)
M2P=$(best_acc sl_v2_M2p_gn_vec)
DIFF=$(python -c "print('%.5f'%($M2-$M2P))")
echo "M2(BN+vec)=$M2  M2'(GN+vec)=$M2P  diff(M2-M2')=$DIFF"
INTER=$(python -c "d=$M2-$M2P; print('strong' if d>0.005 else ('weak' if d>0 else 'none'))")
case "$INTER" in
  strong) echo "[诊断] BN+vec 反超 GN+vec >0.5pp => 存在 norm×vec 正向交互(汇报标注);按去噪微改动纪律仍不跑 M3-BN" ;;
  weak)   echo "[诊断] BN+vec 仅微超 GN+vec(<=0.5pp,噪声级)=> 无实质 norm×vec 交互" ;;
  *)      echo "[诊断] GN+vec 不差于 BN+vec => norm 优劣未被 vec 翻转,GN 确认更优" ;;
esac

echo "===[B2] M3' (v2, B, GN, +vec, +去噪;唯一 M3,不跑 BN 版) $(date +%T)==="
python src/train_sl_v2.py --config B --norm gn --epochs 20 --workers "$W" \
    --data "$DATA" --run-dir "$CKPT/sl_v2_M3_gn_denoise" --exclude-asym

echo "===[去噪净增量] M3'-M2' $(date +%T)==="
M3P=$(best_acc sl_v2_M3_gn_denoise)
DDIFF=$(python -c "print('%.5f'%($M3P-$M2P))")
echo "M3'(GN+vec+去噪)=$M3P  M2'(GN+vec)=$M2P  去噪净增量(M3'-M2')=$DDIFF"
SIG=$(python -c "print('sig' if abs($M3P-$M2P)>=0.003 else 'noise')")
if [ "$SIG" = "noise" ]; then
    echo "[去噪] 增益 <0.3pp,落在噪声内 => 基座取 M2'/M3' 最高者均可,不必纠结"
else
    echo "[去噪] 增益 >=0.3pp,显著"
fi

echo "===[B3] 选最优 v2:{M0,M1,M2,M2',M3'} 取 val 最高 $(date +%T)==="
PICK=$(python - <<'PY'
import os, csv
ck = '/root/autodl-tmp/mahjong/checkpoints'
def best_acc(d):
    p = os.path.join(ck, d, 'epoch_log.csv')
    if not os.path.exists(p):
        return -1.0
    m = -1.0
    with open(p) as f:
        for row in csv.DictReader(f):
            m = max(m, float(row['val_acc']))
    return m
# M0/M1 是 v1(无 vec 分支),其余 v2。平局按 path 排序,GN 版路径靠后胜出,倾向 GN。
cand_defs = (('sl_v2_M0_gn', 'v1'),
             ('sl_v2_M1_bn', 'v1'),
             ('sl_v2_M2_vec', 'v2'),
             ('sl_v2_M2p_gn_vec', 'v2'),
             ('sl_v2_M3_gn_denoise', 'v2'))
cands = []
for d, typ in cand_defs:
    a = best_acc(d)
    if a >= 0:
        cands.append((a, typ, os.path.join(ck, d, 'best.pt')))
cands.sort()
if cands:
    a, typ, p = cands[-1]
    print('%s %s %.5f' % (typ, p, a))
PY
)
echo "best v2 pick = $PICK"
BTYPE=$(echo "$PICK" | awk '{print $1}')
BPATH=$(echo "$PICK" | awk '{print $2}')
if [ "$BTYPE" = "v2" ]; then CAND="nnv2:$BPATH"; else CAND="nn:$BPATH"; fi
echo "arena candidate = $CAND"

echo "===[B4] arena:best SL-v2 vs sl_champion & vs rl_v1_B(各 500 组)$(date +%T)==="
python src/arena/run_arena.py --candidate "$CAND" --opponent "nn:$SL" \
    --groups 500 --workers "$W" --seed 80613 --out results/sl_v2_vs_sl.md
python src/arena/run_arena.py --candidate "$CAND" --opponent "nn:$RL" \
    --groups 500 --workers "$W" --seed 80614 --out results/sl_v2_vs_rl.md

echo "GEN2_STAGE1_DONE $(date +%T) best=$CAND"
