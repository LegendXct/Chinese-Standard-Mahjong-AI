# -*- coding: utf-8 -*-
"""层级头预研 Q1:M5 在验证集上的 masked top-1 误差【按动作类型】分解(纯前向,不训练)。

判别层级(action/tile 两段)头对【我们】有没有用:M5 的错误主要是"选错动作类型"
(过/吃/碰/杠/胡 — action 段)还是"类型对、选错具体牌"(打哪张 — tile 段)?
若错误绝大多数在 tile 段,层级头收益空间在 tile;若 M5 两段都已很准,收益有限。

口径:复用 data_v1 的 obs/mask/vec/act + split val_idx;normalize_obs 同训练;
masked argmax(与 arena/部署一致)。纯 CPU 推理,几分钟。
用法:python scripts/m5_error_decomp.py
"""
import os
import sys
import json

os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS') or '1'
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, '..', 'src')))
import torch  # noqa: E402
from model_v2 import MahjongNetV2  # noqa: E402
from dataset_v2 import normalize_obs  # noqa: E402

ROOT = '/root/autodl-tmp/mahjong/data_v1'
M5 = '/root/autodl-tmp/mahjong/checkpoints/sl_v2_M5_gelu/best.pt'
NAMES = ['Pass', 'Hu', 'Play', 'Chi', 'Peng', 'Gang', 'AnGang', 'BuGang']
# 动作区间(feature_v1.OFFSET_ACT):Pass0 Hu1 Play2-35 Chi36-98 Peng99-132
# Gang133-166 AnGang167-200 BuGang201-234
LO = np.array([0, 1, 2, 36, 99, 133, 167, 201])
HI = np.array([1, 2, 36, 99, 133, 167, 201, 235])


def type_of(a):
    t = np.zeros(len(a), dtype=np.int64)
    for ti in range(8):
        t[(a >= LO[ti]) & (a < HI[ti])] = ti
    return t


def main():
    torch.set_num_threads(20)
    obs = np.load(ROOT + '/obs.dat', mmap_mode='r')
    mask = np.load(ROOT + '/mask.dat', mmap_mode='r')
    act = np.load(ROOT + '/act.dat', mmap_mode='r')
    vec = np.load(ROOT + '/vec.dat', mmap_mode='r')
    val = np.load(ROOT + '/split.npz')['val_idx']
    val = np.sort(val)        # 排序 → memmap 顺序读,快
    N = len(val)
    net, _ = MahjongNetV2.load(M5)
    net.eval()

    true_a = np.asarray(act[val]).astype(np.int64)
    pred_a = np.empty(N, dtype=np.int64)
    nlegal = np.empty(N, dtype=np.int64)
    B = 8192
    with torch.no_grad():
        for i in range(0, N, B):
            idx = val[i:i + B]
            o = normalize_obs(torch.from_numpy(np.ascontiguousarray(obs[idx])))
            v = torch.from_numpy(np.ascontiguousarray(vec[idx]).astype(np.float32))
            m = np.ascontiguousarray(mask[idx]).astype(np.float32)
            logits, _ = net(o, v)
            logits = logits + torch.from_numpy(1.0 - m) * (-1e9)
            pred_a[i:i + B] = logits.argmax(1).numpy()
            nlegal[i:i + B] = m.sum(1).astype(np.int64)
            if i % (B * 8) == 0:
                print('  %d/%d' % (i, N), flush=True)

    tt = type_of(true_a)
    pt = type_of(pred_a)
    correct = (pred_a == true_a)
    type_correct = (pt == tt)
    overall = correct.mean()

    # 仅统计"自由决策点"(合法动作>1);唯一合法的不算决策
    free = nlegal > 1

    print('\n==== M5 验证集 masked top-1 误差分解 ====', flush=True)
    print('总样本 %d;overall top-1 = %.4f(应≈0.8955 校验);自由决策点占比 %.3f'
          % (N, overall, free.mean()), flush=True)
    print('\n按真实动作类型(全样本):', flush=True)
    print('%-8s %9s %7s %8s %9s %10s' % ('type', 'count', 'ratio', 'top1acc', 'typeacc', 'tile|type'))
    rows = {}
    for ti, nm in enumerate(NAMES):
        sel = tt == ti
        c = int(sel.sum())
        if c == 0:
            continue
        a1 = correct[sel].mean()
        ta = type_correct[sel].mean()
        # 给定类型预测正确时,具体牌/参数也正确的比例
        both = sel & type_correct
        tile_given = correct[both].mean() if both.sum() else float('nan')
        rows[nm] = {'count': c, 'ratio': round(c / N, 4), 'top1': round(float(a1), 4),
                    'typeacc': round(float(ta), 4), 'tile_given_type': round(float(tile_given), 4)}
        print('%-8s %9d %7.4f %8.4f %9.4f %10.4f' % (nm, c, c / N, a1, ta, tile_given), flush=True)

    # 误差归因:在所有错误里,多少是"类型对、选错牌(tile 段)" vs "选错类型(action 段)"
    err = ~correct
    err_free = err & free
    wrong_tile = err & type_correct          # 类型对、具体不对
    wrong_type = err & (~type_correct)        # 类型错
    def frac(x, base):
        return round(float(x.sum()) / max(int(base.sum()), 1), 4)
    print('\n误差归因(全部错误 %d):' % int(err.sum()), flush=True)
    print('  类型对·选错牌(tile 段) 占错误 %.4f' % frac(wrong_tile, err), flush=True)
    print('  选错类型(action 段)    占错误 %.4f' % frac(wrong_type, err), flush=True)
    print('误差归因(仅自由决策点错误 %d):' % int(err_free.sum()), flush=True)
    print('  类型对·选错牌 占 %.4f / 选错类型 占 %.4f'
          % (frac(wrong_tile & free, err_free), frac(wrong_type & free, err_free)), flush=True)

    # Play(打牌)单列:它占决策 78%,是 tile 段主战场
    play = tt == 2
    print('\nPlay(弃牌)专项:count %d,top1(=选对弃牌)%.4f,误预测为非Play(类型漏)%.4f'
          % (int(play.sum()), float(correct[play].mean()),
             float((pt[play] != 2).mean())), flush=True)

    out = {'overall_top1': round(float(overall), 4), 'n_val': N,
           'free_ratio': round(float(free.mean()), 4),
           'per_type': rows,
           'errors_wrong_tile_frac': frac(wrong_tile, err),
           'errors_wrong_type_frac': frac(wrong_type, err),
           'errors_free_wrong_tile_frac': frac(wrong_tile & free, err_free),
           'errors_free_wrong_type_frac': frac(wrong_type & free, err_free),
           'play_top1': round(float(correct[play].mean()), 4),
           'play_mispredicted_as_nonplay': round(float((pt[play] != 2).mean()), 4)}
    json.dump(out, open('/root/autodl-tmp/mahjong/results/m5_error_decomp.json', 'w'),
              ensure_ascii=False, indent=1)
    print('\nM5_ERROR_DECOMP_DONE', flush=True)


if __name__ == '__main__':
    main()
