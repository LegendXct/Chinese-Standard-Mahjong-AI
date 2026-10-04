"""动作索引对拍:原样运行 baseline/SL/preprocess.py(sample.txt 16 局),
与 v1 管线(replay_driver + FeatureAgentV1 + 同样过滤)输出的 act/mask 逐一比对。"""
import os
import shutil
import subprocess
import sys

import numpy as np

from conftest import ROOT, SAMPLE_TXT
from feature_v1 import FeatureAgentV1
from replay_driver import iter_matches, replay_match


def _run_baseline_preprocess(tmpdir):
    """在临时目录原样跑 baseline preprocess(sample.txt 当 data.txt)。"""
    datadir = os.path.join(tmpdir, 'data')
    os.makedirs(datadir, exist_ok=True)
    shutil.copy(SAMPLE_TXT, os.path.join(datadir, 'data.txt'))
    env = dict(os.environ)
    env['PYTHONPATH'] = os.path.join(ROOT, 'baseline', 'SL')
    subprocess.run([sys.executable, os.path.join(ROOT, 'baseline', 'SL', 'preprocess.py')],
                   cwd=tmpdir, env=env, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # 3.6 无 capture_output
    return datadir


def test_act_and_mask_match_baseline(tmp_path):
    datadir = _run_baseline_preprocess(str(tmp_path))
    n_games = 0
    for gid, (match_id, lines) in enumerate(iter_matches(SAMPLE_TXT)):
        base = np.load(os.path.join(datadir, '%d.npz' % gid))
        r = replay_match(match_id, lines, FeatureAgentV1)
        acts, masks = [], []
        for seat in range(4):
            for o, a in zip(r['obs'][seat], r['act'][seat]):
                if o['action_mask'].sum() > 1:
                    acts.append(a)
                    masks.append(o['action_mask'])
        acts = np.array(acts)
        masks = np.stack(masks)
        assert acts.shape == base['act'].shape, \
            'match %s: sample count differs %s vs %s' % (match_id, acts.shape, base['act'].shape)
        assert (acts == base['act']).all(), \
            'match %s: act mismatch at %s' % (match_id, np.flatnonzero(acts != base['act'])[:10])
        assert (masks == base['mask']).all(), \
            'match %s: mask mismatch' % match_id
        n_games += 1
    assert n_games == 16
