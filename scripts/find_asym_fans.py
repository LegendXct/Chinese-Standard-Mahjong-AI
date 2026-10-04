# -*- coding: utf-8 -*-
"""扫 data.txt 标记含"绿一色/推不倒"的对局(花色不对称番种,花色置换会改变番数,
增广时应排除)。gid = 对局在文件中的出现序(与 preprocess_v1 的 meta['gid'] 同序)。

输出 <data_root>/asym_fan_gids.json(排序的 gid 列表)。
用法:python scripts/find_asym_fans.py --data-txt .../data.txt --out .../data_v1
对应 Arthals valid.py 的 cannot_enhance_matches。
"""
import argparse
import json
import os
import re

PATTERN = re.compile('绿一色|推不倒')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-txt', default='/root/autodl-tmp/mahjong/baseline/SL/data/data.txt')
    ap.add_argument('--out', default='/root/autodl-tmp/mahjong/data_v1')
    args = ap.parse_args()

    gid = -1
    asym = []
    n_match = 0
    with open(args.data_txt, encoding='utf-8') as f:
        for line in f:
            if line.startswith('Match'):
                gid += 1
                n_match += 1
            elif line.startswith('Fan ') and PATTERN.search(line):
                asym.append(gid)
    asym = sorted(set(asym))
    out_path = os.path.join(args.out, 'asym_fan_gids.json')
    with open(out_path, 'w') as f:
        json.dump(asym, f)
    print('matches scanned: %d; asym(绿一色/推不倒) matches: %d (%.2f%%)' % (
        n_match, len(asym), 100.0 * len(asym) / max(n_match, 1)), flush=True)
    print('written %s' % out_path, flush=True)


if __name__ == '__main__':
    main()
