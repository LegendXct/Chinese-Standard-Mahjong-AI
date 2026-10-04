"""冒烟测试 B:baseline/RL 的 CNNModel 在 GPU 上做一次前向,验证 sm_120 兼容性。

在服务器项目根目录运行:python src/smoke_gpu_forward.py
"""
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'baseline', 'RL'))
from model import CNNModel  # noqa: E402

print('torch', torch.__version__, 'cuda', torch.version.cuda,
      'device', torch.cuda.get_device_name(0),
      'capability', torch.cuda.get_device_capability(0))

model = CNNModel().cuda().eval()
obs = torch.randint(0, 2, (128, 6, 4, 9), dtype=torch.int8, device='cuda')
mask = torch.ones(128, 235, device='cuda')

with torch.no_grad():
    # 第一次前向含 CUDA kernel 编译/加载开销,单独计时
    torch.cuda.synchronize()
    t0 = time.time()
    logits, value = model({'observation': obs, 'action_mask': mask})
    torch.cuda.synchronize()
    t_first = time.time() - t0

    t0 = time.time()
    for _ in range(100):
        logits, value = model({'observation': obs, 'action_mask': mask})
    torch.cuda.synchronize()
    t_avg = (time.time() - t0) / 100

print('forward ok: logits %s value %s' % (tuple(logits.shape), tuple(value.shape)))
print('first forward: %.1f ms, steady-state avg: %.2f ms (batch=128)' % (t_first * 1000, t_avg * 1000))
