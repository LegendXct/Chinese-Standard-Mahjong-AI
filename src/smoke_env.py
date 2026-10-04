"""冒烟测试 C:用随机合法动作在 CPU 上完整跑 2 局,验证环境流转与算番器。

在服务器项目根目录运行:python src/smoke_env.py
"""
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'baseline', 'RL'))
from env import MahjongGBEnv  # noqa: E402
from feature import FeatureAgent  # noqa: E402

random.seed(42)
env = MahjongGBEnv(config={'agent_clz': FeatureAgent})

for episode in range(2):
    t0 = time.time()
    obs = env.reset()
    done = False
    steps = 0
    rewards = {}
    while not done:
        action_dict = {}
        for name, o in obs.items():
            valid = np.flatnonzero(o['action_mask'])
            assert len(valid) > 0, 'empty action mask for %s at step %d' % (name, steps)
            action_dict[name] = int(random.choice(valid))
        obs, rewards, done = env.step(action_dict)
        steps += 1
    if all(r == 0 for r in rewards.values()):
        outcome = 'Huang(荒庄)'
    elif min(rewards.values()) == -30:
        outcome = 'Invalid(非法动作)'
    else:
        winner = max(rewards, key=rewards.get)
        outcome = 'Hu(%s 胡牌)' % winner
    print('episode %d: %s, steps=%d, time=%.2fs, rewards=%s' % (
        episode, outcome, steps, time.time() - t0, rewards))

print('env smoke test ok')
