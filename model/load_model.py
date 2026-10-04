# -*- coding: utf-8 -*-
"""M5 权重加载与推理自检。

验证两件事:
  1) model/sl_v2_m5_gelu.pt 能被 MahjongNetV2.load 正常加载并前向(标准 fp32 路径);
  2) 部署用的 FP16ResidentNetV2(fp16 常驻、逐算子转 fp32)与 fp32 路径输出一致
     —— 即 Botzone 线上 bot 实际走的推理路径可用。

用法(仓库根目录下):
    python model/load_model.py                    # 默认读 model/sl_v2_m5_gelu.pt
    python model/load_model.py <path/to/ckpt.pt>  # 或指定任意 v2 checkpoint
权重不随仓库发布,需先按 README 训练得到。
不依赖 PyMahjongGB / 游戏引擎,只需 numpy + torch(>=1.4)。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np
import torch

from model_v2 import MahjongNetV2, FP16ResidentNetV2

WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'sl_v2_m5_gelu.pt')


def main():
    print('torch:', torch.__version__)
    print('weights:', WEIGHTS)

    # 1) 标准加载 + 前向
    model, payload = MahjongNetV2.load(WEIGHTS)
    model.eval()
    cfg = payload['config']
    print('config:', cfg)
    print('params: %.2fM' % (model.n_params() / 1e6))
    if 'val_acc' in payload:
        print('val_acc (训练时记录): %.5f' % payload['val_acc'])

    # 造一个合法形状的随机观测(0/1)与标量向量
    rng = np.random.RandomState(0)
    obs = torch.from_numpy((rng.rand(4, 102, 4, 9) > 0.7).astype('float32'))
    vec = torch.from_numpy(rng.rand(4, cfg['vec_dim']).astype('float32'))
    with torch.no_grad():
        logits_fp32, value_fp32 = model(obs, vec)
    print('fp32 forward OK: logits', tuple(logits_fp32.shape),
          'value', tuple(value_fp32.shape))

    # 2) 部署路径(fp16 常驻)一致性
    resident = FP16ResidentNetV2(payload['state_dict'], cfg)
    with torch.no_grad():
        logits_fp16, _ = resident(obs, vec)
    same = int((logits_fp32.argmax(-1) == logits_fp16.argmax(-1)).all())
    max_abs = float((logits_fp32 - logits_fp16).abs().max())
    print('fp16_resident argmax 与 fp32 一致:', bool(same),
          '(logits 最大绝对差 %.3e)' % max_abs)

    ok = (logits_fp32.shape[-1] == 235) and same
    print('\nSELF-CHECK:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
