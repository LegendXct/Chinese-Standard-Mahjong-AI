#!/usr/bin/env bash
# botzone-compat 环境验证:pytest 快档 + 跨环境预处理逐元素对拍
set -uo pipefail
cd /root/autodl-tmp/mahjong
PY=/root/miniconda3/envs/botzone-compat/bin/python

echo "===pytest-compat==="
AUG_GAMES=8 $PY -m pytest tests/ -x -q 2>&1 | tail -3

echo "===preprocess-both==="
rm -rf tmp/pp_compat tmp/pp_main
$PY src/preprocess_v1.py --data baseline/SL/data/sample.txt --out tmp/pp_compat --workers 4 > /dev/null 2>&1 && echo compat-pp-ok
python src/preprocess_v1.py --data baseline/SL/data/sample.txt --out tmp/pp_main --workers 4 > /dev/null 2>&1 && echo main-pp-ok

python - <<'EOF'
import numpy as np
for f in ('obs.dat', 'mask.dat', 'act.dat'):
    a = np.load('tmp/pp_main/' + f, mmap_mode='r')
    b = np.load('tmp/pp_compat/' + f, mmap_mode='r')
    assert a.shape == b.shape and np.array_equal(a, b), f
print('CROSS-ENV ELEMENTWISE EQUAL')
EOF
echo COMPAT_TESTS_DONE
