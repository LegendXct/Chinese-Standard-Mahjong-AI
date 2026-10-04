"""全量预处理:data.txt -> memmap 数据集(特征 v1)。

按 Match 字节偏移把文件切成 K 份,多进程并行重放,worker 写 part 文件,
主进程顺序合并进最终 memmap(边合并边删 part,峰值磁盘 ≈ 最终大小 + 单个 part)。

输出(--out 目录):
  obs.dat   uint8 (N, C, 4, 9)
  mask.dat  uint8 (N, 235)
  act.dat   int16 (N,)
  meta.npz  每样本 game_id/seat/step/score/has_winner/winner_rel/fan + 每局 scores/winner/fan/match_id/n_samples
  split.npz 按局 95/5 的 train/val 局号与样本索引(固定种子)
  info.json N、C、版本、动作分布等
  violations.json 重放合法性违规(目标为空)

用法(服务器,项目根目录):
  python src/preprocess_v1.py --data baseline/SL/data/data.txt --out /root/autodl-tmp/mahjong/data_v1 --workers 128
  python src/preprocess_v1.py --data baseline/SL/data/sample.txt --out tmp/data_v1_sample --workers 4
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_v1 import FeatureAgentV1, NUM_CHANNELS, FEATURE_VERSION  # noqa: E402
from replay_driver import iter_matches, replay_match, check_legality  # noqa: E402

OFFSET_ACT = FeatureAgentV1.OFFSET_ACT
SPLIT_SEED = 20260610
VAL_RATIO = 0.05


def index_match_offsets(path):
    """返回每局 Match 行的字节偏移列表 + 文件总长。"""
    offsets = []
    pos = 0
    with open(path, 'rb') as f:
        for line in f:
            if line.startswith(b'Match'):
                offsets.append(pos)
            pos += len(line)
    return offsets, pos


def _iter_matches_in_slice(path, start, end):
    with open(path, 'rb') as f:
        f.seek(start)
        text = f.read(end - start).decode('utf-8')
    match_id = None
    lines = []
    for line in text.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == 'Match':
            if match_id is not None:
                yield match_id, lines
            match_id = t[1]
            lines = []
        else:
            lines.append(line)
    if match_id is not None:
        yield match_id, lines


def _part_valid(part, expect_games):
    """--resume 完整性校验:三个文件齐全、可加载、数量自洽。
    np.save 是 chunk 完成时一次性写入,截断文件会在 np.load/形状校验时暴露。"""
    try:
        po = np.load(part + '_obs.npy', mmap_mode='r')
        pm = np.load(part + '_mask.npy', mmap_mode='r')
        pz = np.load(part + '_meta.npz', allow_pickle=False)
        n = len(pz['act'])
        ok = (po.shape == (n, NUM_CHANNELS, 4, 9) and pm.shape == (n, 235)
              and len(pz['gid']) == n and len(pz['seat']) == n
              and len(pz['g_mid']) == expect_games
              and int(pz['g_nsamples'].sum()) == n)
        del po, pm
        return ok
    except Exception:
        return False


def process_chunk(args):
    """worker:处理 [start,end) 字节区间的对局,写 part 文件。
    resume 时对已有合法 part 直接复用(违规列表无法恢复,由独立的
    replay_check 步骤兜底)。"""
    chunk_id, path, start, end, first_gid, expect_games, resume, out_dir = args
    import resource
    part = os.path.join(out_dir, 'parts', 'p%04d' % chunk_id)
    if resume and _part_valid(part, expect_games):
        pz = np.load(part + '_meta.npz', allow_pickle=False)
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        return chunk_id, len(pz['act']), len(pz['g_mid']), [], rss_mb, True
    obs_list, mask_list, act_list = [], [], []
    s_gid, s_seat, s_step, s_score, s_haswin, s_winrel, s_fan = [], [], [], [], [], [], []
    g_scores, g_winner, g_fan, g_mid, g_nsamples = [], [], [], [], []
    violations = []

    gid = first_gid
    for match_id, lines in _iter_matches_in_slice(path, start, end):
        r = replay_match(match_id, lines, FeatureAgentV1)
        for v in check_legality(r):
            violations.append({'game': gid, 'match_id': match_id,
                               'seat': v[0], 'step': v[1], 'act': v[2], 'mask_sum': v[3]})
        n_game = 0
        for seat in range(4):
            for step, (o, a) in enumerate(zip(r['obs'][seat], r['act'][seat])):
                if o['action_mask'].sum() > 1:  # 与 baseline 相同的过滤
                    obs_list.append(o['observation'])
                    mask_list.append(o['action_mask'])
                    act_list.append(a)
                    s_gid.append(gid)
                    s_seat.append(seat)
                    s_step.append(step)
                    s_score.append(r['scores'][seat])
                    s_haswin.append(1 if r['winner'] >= 0 else 0)
                    s_winrel.append((r['winner'] - seat) % 4 if r['winner'] >= 0 else -1)
                    s_fan.append(r['fan'])
                    n_game += 1
        g_scores.append(r['scores'])
        g_winner.append(r['winner'])
        g_fan.append(r['fan'])
        g_mid.append(r['match_id'])
        g_nsamples.append(n_game)
        gid += 1

    np.save(part + '_obs.npy', np.stack(obs_list).astype(np.uint8) if obs_list
            else np.zeros((0, NUM_CHANNELS, 4, 9), np.uint8))
    np.save(part + '_mask.npy', np.stack(mask_list).astype(np.uint8) if mask_list
            else np.zeros((0, 235), np.uint8))
    np.savez(part + '_meta.npz',
             act=np.array(act_list, np.int16),
             gid=np.array(s_gid, np.int32), seat=np.array(s_seat, np.int8),
             step=np.array(s_step, np.int16), score=np.array(s_score, np.int16),
             haswin=np.array(s_haswin, np.int8), winrel=np.array(s_winrel, np.int8),
             fan=np.array(s_fan, np.int16),
             g_scores=np.array(g_scores, np.int16).reshape(-1, 4),
             g_winner=np.array(g_winner, np.int8), g_fan=np.array(g_fan, np.int16),
             g_mid=np.array(g_mid), g_nsamples=np.array(g_nsamples, np.int32))
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    return chunk_id, len(act_list), gid - first_gid, violations, rss_mb, False


def action_type_distribution(act):
    bounds = [('Pass', 0, 1), ('Hu', 1, 2), ('Play', 2, 36), ('Chi', 36, 99),
              ('Peng', 99, 133), ('Gang', 133, 167), ('AnGang', 167, 201), ('BuGang', 201, 235)]
    n = len(act)
    return {name: {'count': int(((act >= lo) & (act < hi)).sum()),
                   'ratio': round(float(((act >= lo) & (act < hi)).sum()) / max(n, 1), 6)}
            for name, lo, hi in bounds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--limit', type=int, default=None, help='只处理前 N 局(调试用)')
    ap.add_argument('--resume', action='store_true',
                    help='复用已存在且通过完整性校验的 part 文件')
    ap.add_argument('--tasks-per-child', type=int, default=2,
                    help='worker 处理 N 个 chunk 后重生(清空缓存控内存)')
    ap.add_argument('--chunks', type=int, default=None,
                    help='切块数(默认 workers*4)。--resume 必须与原跑保持一致,否则边界不同无法复用')
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.join(args.out, 'parts'), exist_ok=True)

    offsets, fsize = index_match_offsets(args.data)
    if args.limit:
        offsets = offsets[:args.limit + 1] if args.limit < len(offsets) else offsets
        end_of_data = offsets[args.limit] if args.limit < len(offsets) else fsize
        offsets = offsets[:args.limit]
    else:
        end_of_data = fsize
    n_games = len(offsets)
    print('total matches: %d (index %.1fs)' % (n_games, time.time() - t0))

    n_chunks = min(args.chunks or args.workers * 4, n_games)  # 多倍切块:负载均衡 + 控单 chunk 内存
    bnd = np.linspace(0, n_games, n_chunks + 1).astype(int)
    tasks = []
    for c in range(n_chunks):
        lo, hi = bnd[c], bnd[c + 1]
        if lo == hi:
            continue
        end = offsets[hi] if hi < n_games else end_of_data
        tasks.append((c, args.data, offsets[lo], end, lo, hi - lo, args.resume, args.out))

    t1 = time.time()
    results = []
    n_reused = 0
    with Pool(args.workers, maxtasksperchild=args.tasks_per_child) as pool:
        for r in pool.imap_unordered(process_chunk, tasks):
            results.append(r)
            n_reused += 1 if r[5] else 0
            done = len(results)
            if done % 16 == 0 or done == len(tasks):
                print('chunks %d/%d (reused %d), samples so far %d, peak worker RSS %d MB, %.1fs'
                      % (done, len(tasks), n_reused, sum(x[1] for x in results),
                         max(x[4] for x in results), time.time() - t1), flush=True)
    results.sort(key=lambda x: x[0])
    n_samples = sum(r[1] for r in results)
    all_violations = [v for r in results for v in r[3]]
    print('replay done: %d samples (%d chunks reused), %d violations, peak worker RSS %d MB, %.1fs' % (
        n_samples, n_reused, len(all_violations), max(r[4] for r in results), time.time() - t1))

    # ---- 合并 ----
    t2 = time.time()
    obs_mm = np.lib.format.open_memmap(os.path.join(args.out, 'obs.dat'), mode='w+',
                                       dtype=np.uint8, shape=(n_samples, NUM_CHANNELS, 4, 9))
    mask_mm = np.lib.format.open_memmap(os.path.join(args.out, 'mask.dat'), mode='w+',
                                        dtype=np.uint8, shape=(n_samples, 235))
    act_arr = np.zeros(n_samples, np.int16)
    meta = {k: [] for k in ('gid', 'seat', 'step', 'score', 'haswin', 'winrel', 'fan')}
    gmeta = {k: [] for k in ('g_scores', 'g_winner', 'g_fan', 'g_mid', 'g_nsamples')}
    pos = 0
    for chunk_id, cnt, _, _, _, _ in results:
        part = os.path.join(args.out, 'parts', 'p%04d' % chunk_id)
        po = np.load(part + '_obs.npy', mmap_mode='r')
        pm = np.load(part + '_mask.npy', mmap_mode='r')
        pz = np.load(part + '_meta.npz', allow_pickle=False)
        assert len(po) == cnt
        obs_mm[pos:pos + cnt] = po
        mask_mm[pos:pos + cnt] = pm
        act_arr[pos:pos + cnt] = pz['act']
        for k in meta:
            meta[k].append(pz[k])
        for k in gmeta:
            gmeta[k].append(pz[k])
        pos += cnt
        del po, pm
        for suf in ('_obs.npy', '_mask.npy', '_meta.npz'):
            os.remove(part + suf)
    assert pos == n_samples
    obs_mm.flush()
    mask_mm.flush()
    del obs_mm, mask_mm
    os.rmdir(os.path.join(args.out, 'parts'))

    np.save(os.path.join(args.out, 'act.dat.npy'), act_arr)
    os.replace(os.path.join(args.out, 'act.dat.npy'), os.path.join(args.out, 'act.dat'))
    meta = {k: np.concatenate(v) for k, v in meta.items()}
    gmeta = {k: np.concatenate(v) for k, v in gmeta.items()}
    np.savez(os.path.join(args.out, 'meta.npz'), act=act_arr, **meta, **gmeta)
    print('merge done %.1fs' % (time.time() - t2))

    # ---- 按局切分 train/val ----
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(n_games)
    n_val = int(round(n_games * VAL_RATIO))
    val_games = np.sort(perm[:n_val])
    train_games = np.sort(perm[n_val:])
    is_val_game = np.zeros(n_games, bool)
    is_val_game[val_games] = True
    val_mask = is_val_game[meta['gid']]
    np.savez(os.path.join(args.out, 'split.npz'),
             seed=SPLIT_SEED, train_games=train_games, val_games=val_games,
             train_idx=np.flatnonzero(~val_mask), val_idx=np.flatnonzero(val_mask))

    info = {
        'feature_version': FEATURE_VERSION,
        'channels': NUM_CHANNELS,
        'n_samples': n_samples,
        'n_games': n_games,
        'samples_per_game': round(n_samples / max(n_games, 1), 2),
        'n_train_games': int(len(train_games)), 'n_val_games': int(len(val_games)),
        'n_train_samples': int((~val_mask).sum()), 'n_val_samples': int(val_mask.sum()),
        'n_violations': len(all_violations),
        'action_distribution': action_type_distribution(act_arr),
        'obs_bytes': int(n_samples) * NUM_CHANNELS * 36,
        'elapsed_sec': round(time.time() - t0, 1),
        'workers': args.workers,
    }
    with open(os.path.join(args.out, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(args.out, 'violations.json'), 'w') as f:
        json.dump(all_violations, f, indent=2)
    print(json.dumps(info, indent=2))
    print('total %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
