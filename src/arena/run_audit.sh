#!/usr/bin/env bash
# Arena 尺子审计完整电池 — 25 核(开卡)CPU,argmax+采样,不训练、不动 GPU。
# Phase A 走既有路径;Phase B 用本次新增的可选参数(--temperature/--indep-opp/--opponents/
# --decision-salt),默认不传则老 argmax 1v3 路径字节级不变。各 run 串行(每 run 占满 24 核)。
set -u
cd /root/autodl-tmp/mahjong
OUT=results/arena_audit; mkdir -p "$OUT"
SEED=80640; G=500; W=24
M5="nnv2:checkpoints/sl_v2_M5_gelu/best.pt"
M3="nnv2:checkpoints/sl_v2_M3_gn_denoise/best.pt"
V1="nn:checkpoints/sl_B_gn/best.pt"
RULE="rule"
S3POT="nnv2:checkpoints/rl_s3_potential/last.pt"
S4CMB="nnv2:checkpoints/rl_s4_combo/snap_0308.pt"
HET="rule,${M3},${M5}"          # Test#3a 异质三对手:rule + M3' + M5
GREP="同墙配对差值|每局平均得分|胡牌率|点炮率|胡牌平均番|temperature|elapsed"

run() {  # tag cand opp [extra args...]
  local tag="$1" cand="$2" opp="$3"; shift 3
  echo "===== ${tag} : ${cand} vs ${opp} $* | $(date '+%T') ====="
  python src/arena/run_arena.py --candidate "$cand" --opponent "$opp" \
    --groups $G --seed $SEED --workers $W --out "$OUT/${tag}.md" "$@" 2>&1 | grep -E "$GREP"
  echo
}

echo "################ Phase A:非门控(既有路径,argmax)################"
echo "#### Test #4 标定阶梯 ####"
run t4_M5_vs_M3   "$M5" "$M3"      # 预期复现 +4.42(尺子校验)
run t4_M5_vs_v1   "$M5" "$V1"
run t4_M3_vs_v1   "$M3" "$V1"
run t4_M5_vs_rule "$M5" "$RULE"
run t4_M3_vs_rule "$M3" "$RULE"

echo "#### Test #1 T=0 锚点(RL vs M5,argmax)####"
run t1_s3pot_T0   "$S3POT" "$M5"
run t1_s4combo_T0 "$S4CMB" "$M5"

echo "#### Test #3b 弱/异质单一对手(argmax)####"
run t3b_s3pot_rule  "$S3POT" "$RULE"
run t3b_s3pot_v1    "$S3POT" "$V1"
run t3b_s4combo_rule "$S4CMB" "$RULE"
run t3b_s4combo_v1   "$S4CMB" "$V1"
# M5 vs rule / M5 vs v1 复用 t4_M5_vs_rule / t4_M5_vs_v1

echo "#### Test #2 轨迹偏离(候选 vs 纯 M5,argmax,只读)####"
for tag in s3pot:$S3POT s4combo:$S4CMB; do
  nm="${tag%%:*}"; sp="${tag#*:}"
  echo "----- dev_${nm} -----"
  python src/arena/deviation_probe.py --candidate "$sp" --ref "$M5" \
    --groups $G --seed $SEED --workers $W --out "$OUT/t2_dev_${nm}.json" 2>&1 \
    | grep -vE '"records"' | grep -E 'deviation|paired_diff|diff_|free_|DONE|mean|se|n_'
  echo
done

echo "################ Phase B:门控(新可选参数)################"
echo "#### Test #1 温度对称扫(RL vs M5,--indep-opp,同墙 2 salt)####"
for cd in s3pot:$S3POT s4combo:$S4CMB; do
  nm="${cd%%:*}"; sp="${cd#*:}"
  for T in 0.3 0.6 1.0; do
    for SALT in 0 1; do
      run "t1_${nm}_T${T}_s${SALT}" "$sp" "$M5" --temperature "$T" --indep-opp --decision-salt "$SALT"
    done
  done
done

echo "#### Test #3a 异质三对手场(argmax,rule+M3'+M5)####"
run t3a_s3pot_het   "$S3POT" "$M5" --opponents "$HET"
run t3a_s4combo_het "$S4CMB" "$M5" --opponents "$HET"
run t3a_M5_het      "$M5"    "$M5" --opponents "$HET"   # M5 自己在同一异质场(对照基准)

echo "===== AUDIT_BATTERY_DONE $(date '+%T') ====="
