"""增广一致性:日志级置换(直接改写牌谱文本中的 W/T/B 牌面后重放)
与置换表变换后的 (obs, mask, act) 必须逐元素一致,6 种排列全测。

唯一允许的例外:mask 的 Hu 位(索引 1)。绿一色/推不倒等番种与具体花色绑定,
花色置换后番数可能跨越 8 番起胡线,导致 Hu 合法位翻转。这是规则的真实性质
而非实现 bug(已人工核验案例:推不倒 8 番在条筒互换后消失)。
此类差异单独计数上报,其余任何差异都视为失败。

局数由环境变量 AUG_GAMES 控制(默认 24;服务器全量校验用 200,
从 data.txt 固定种子随机抽样)。
"""
import os
import random

import numpy as np
import pytest

from conftest import SAMPLE_TXT, DATA_TXT
from feature_v1 import FeatureAgentV1
from replay_driver import iter_matches, replay_match
from dataset_v1 import SUIT_PERMS, augment_sample

AUG_GAMES = int(os.environ.get('AUG_GAMES', 24))
AUG_SEED = 12345


def _permute_log(lines, sigma):
    out = []
    for line in lines:
        toks = []
        for tok in line.split():
            if len(tok) == 2 and tok[0] in 'WTB' and tok[1].isdigit():
                toks.append('WTB'[sigma['WTB'.index(tok[0])]] + tok[1])
            else:
                toks.append(tok)
        out.append(' '.join(toks))
    return out


def _sample_games():
    if os.path.exists(DATA_TXT) and AUG_GAMES > 16:
        # 先数局数再固定种子抽样,只重放被抽中的局
        all_games = []
        for mid, lines in iter_matches(DATA_TXT):
            all_games.append((mid, lines))
        rng = random.Random(AUG_SEED)
        return rng.sample(all_games, AUG_GAMES)
    return [(mid, lines) for mid, lines in iter_matches(SAMPLE_TXT)]


def _flatten(result):
    """对齐展开为 [(seat, step, obs, mask, act)](未过滤,含全部决策点)。"""
    flat = []
    for seat in range(4):
        for step, (o, a) in enumerate(zip(result['obs'][seat], result['act'][seat])):
            flat.append((seat, step, o['observation'], o['action_mask'], a))
    return flat


@pytest.mark.parametrize('k', range(1, 6))
def test_log_level_permutation(k):
    sigma = SUIT_PERMS[k]
    games = _sample_games()
    n_obs = 0
    hu_bit_flips = 0
    for mid, lines in games:
        orig = _flatten(replay_match(mid, lines, FeatureAgentV1))
        perm = _flatten(replay_match(mid, _permute_log(lines, sigma), FeatureAgentV1))
        assert len(orig) == len(perm), 'match %s k=%d: decision count differs' % (mid, k)
        for (s1, t1, o1, m1, a1), (s2, t2, o2, m2, a2) in zip(orig, perm):
            assert (s1, t1) == (s2, t2)
            ao, am, aa = augment_sample(o1, m1, a1, k)
            ctx = 'match %s k=%d seat %d step %d' % (mid, k, s1, t1)
            assert aa == a2, '%s: act %d->%d expected %d' % (ctx, a1, aa, a2)
            if not (am == m2).all():
                d = np.flatnonzero(am != m2)
                if list(d) == [1]:  # 仅 Hu 位:花色绑定番种(绿一色/推不倒)的已知例外
                    hu_bit_flips += 1
                else:
                    raise AssertionError('%s: mask differs at %s' % (ctx, d[:10]))
            if not (ao == o2).all():
                d = np.argwhere(ao != o2)
                raise AssertionError('%s: obs differs at channels %s' % (ctx, np.unique(d[:, 0])[:10]))
            n_obs += 1
    assert n_obs > 0
    print('k=%d: %d games, %d decisions OK, hu-bit flips (suit-bound fans): %d'
          % (k, len(games), n_obs, hu_bit_flips))
    # 该例外必须极其罕见(<0.1% 决策点),否则视为异常
    assert hu_bit_flips < n_obs * 0.001, 'too many hu-bit flips: %d/%d' % (hu_bit_flips, n_obs)
