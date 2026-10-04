#!/usr/bin/env bash
# 测量工具(第二批"尺子"):对 ckpt 跑完整 arena 评测,打印每对手 pair/hu/点炮
#   + 胜局平均番(arena 口径,非自对弈失真)+ 点炮分番(平均番、≥16番点炮率)
#   + 竞争子集 {M5,bighand,def} 聚合基线。与 batch-2 训练评测同口径。需 GPU。
# 用法:CKPT=<path> NGRP=<n> bash scripts/measure_eval.sh   (默认 rl_v2c eval-best, 500 组)
# 注意:变量名勿用 GROUPS——它是 bash 内置只读数组(用户组 GID),赋值无效,root 下读为 0。
set -u
cd /root/autodl-tmp/mahjong
export OMP_NUM_THREADS=1
PY=/root/miniconda3/bin/python
CKPT=${CKPT:-/root/autodl-tmp/mahjong/checkpoints/rl_v2c_evalbest_0345.pt}
NGRP=${NGRP:-500}
echo "[measure-sh] ckpt=$CKPT groups=$NGRP"
$PY src/rl/train_rl.py --mode eval --set init_model=$CKPT run_dir=/root/autodl-tmp/mahjong/checkpoints/measure_tmp n_env_workers=2 n_eval_workers=12 eval_full_groups=$NGRP
