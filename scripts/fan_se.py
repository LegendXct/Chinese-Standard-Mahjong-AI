"""从 arena json 计算候选胜局平均番 + SE,做 bighand vs baseline 的显著性判定。"""
import json
import math
import os
import sys

ROOT = '/root/autodl-tmp/mahjong'


def stat(tag):
    d = json.load(open(os.path.join(ROOT, 'results', '%s.json' % tag)))
    fans = [r['fan'] for r in d['records'] if r['hu'] and r['fan'] > 0]
    n = len(fans)
    m = sum(fans) / n
    se = (sum((x - m) ** 2 for x in fans) / (n - 1)) ** 0.5 / math.sqrt(n)
    return n, m, se


nb, mb, seb = stat('spar_baseline')
nh, mh, seh = stat(sys.argv[1] if len(sys.argv) > 1 else 'spar_bighand_vs_m5')
diff = mh - mb
sed = math.sqrt(seb ** 2 + seh ** 2)
print('baseline: n_win=%d fan=%.3f SE=%.3f' % (nb, mb, seb))
print('bighand : n_win=%d fan=%.3f SE=%.3f' % (nh, mh, seh))
print('diff=%+.3f  SE_diff=%.3f  diff/SE=%.2f  ACCEPT(>1SE)=%s'
      % (diff, sed, diff / sed, diff > sed))
