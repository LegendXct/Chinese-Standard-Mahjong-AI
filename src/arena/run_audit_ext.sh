#!/usr/bin/env bash
# 审计扩展(完整严格版,接 CORE1 之后)。跳过 CORE1 已成功的 3 项
# (t4_M5_vs_M3 / t1_s3pot_T0 / t1_s4combo_T0);dev_s3pot 在 core1 失败,这里重跑。
# 顺序按决定性排:Test#2 偏离 → Test#1 温度扫 → Test#3a 异质场 → Test#4 阶梯余项 → Test#3b。
# 各 run 串行(每 run 占满 24 核)。
set -u
export OMP_NUM_THREADS=1   # 登录环境里为空 → libgomp 报错;单线程 CPU 推理足够
cd /root/autodl-tmp/mahjong
OUT=results/arena_audit; mkdir -p "$OUT" logs
SEED=80640; G=500; W=24
M5="nnv2:checkpoints/sl_v2_M5_gelu/best.pt"
M3="nnv2:checkpoints/sl_v2_M3_gn_denoise/best.pt"
V1="nn:checkpoints/sl_B_gn/best.pt"
RULE="rule"
S3POT="nnv2:checkpoints/rl_s3_potential/last.pt"
S4CMB="nnv2:checkpoints/rl_s4_combo/snap_0308.pt"
HET="rule,${M3},${M5}"
GREP="同墙配对差值|每局平均得分|胡牌率|点炮率|胡牌平均番|temperature|elapsed"
L=logs/audit_ext.log
echo "===== EXT START $(date '+%F %T') =====" | tee "$L"

run() {  # tag cand opp [extra...]
  local tag="$1" cand="$2" opp="$3"; shift 3
  echo "===== ${tag} : ${cand} vs ${opp} $* | $(date '+%T') =====" | tee -a "$L"
  python src/arena/run_arena.py --candidate "$cand" --opponent "$opp" \
    --groups $G --seed $SEED --workers $W --out "$OUT/${tag}.md" "$@" 2>&1 | grep -E "$GREP" | tee -a "$L"
  echo | tee -a "$L"
}
dev() {  # tag cand
  echo "===== t2_dev_$1 : $2 vs 纯M5 @500 | $(date '+%T') =====" | tee -a "$L"
  python src/arena/deviation_probe.py --candidate "$2" --ref "$M5" \
    --groups $G --seed $SEED --workers $W --out "$OUT/t2_dev_$1.json" 2>&1 \
    | grep -vE '"records"' | tee -a "$L"
  echo | tee -a "$L"
}

echo "######## Test#2 轨迹偏离(只读,有效样本量)########" | tee -a "$L"
dev s3pot   "$S3POT"
dev s4combo "$S4CMB"

echo "######## Test#1 完整温度扫(RL vs M5,--indep-opp,@500,2 salt)########" | tee -a "$L"
for cd in s3pot:$S3POT s4combo:$S4CMB; do
  nm="${cd%%:*}"; sp="${cd#*:}"
  for T in 0.3 0.6 1.0; do
    for SALT in 0 1; do
      run "t1_${nm}_T${T}_s${SALT}" "$sp" "$M5" --temperature "$T" --indep-opp --decision-salt "$SALT"
    done
  done
done

echo "######## Test#3a 异质三对手场(argmax,rule+M3'+M5,@500)########" | tee -a "$L"
run t3a_s3pot_het   "$S3POT" "$M5" --opponents "$HET"
run t3a_s4combo_het "$S4CMB" "$M5" --opponents "$HET"
run t3a_M5_het      "$M5"    "$M5" --opponents "$HET"

echo "######## Test#4 标定阶梯余项 ########" | tee -a "$L"
run t4_M5_vs_v1   "$M5" "$V1"
run t4_M3_vs_v1   "$M3" "$V1"
run t4_M5_vs_rule "$M5" "$RULE"
run t4_M3_vs_rule "$M3" "$RULE"

echo "######## Test#3b 弱/异质单一对手 ########" | tee -a "$L"
run t3b_s3pot_rule   "$S3POT" "$RULE"
run t3b_s3pot_v1     "$S3POT" "$V1"
run t3b_s4combo_rule "$S4CMB" "$RULE"
run t3b_s4combo_v1   "$S4CMB" "$V1"

echo "===== EXT DONE $(date '+%F %T') =====" | tee -a "$L"
