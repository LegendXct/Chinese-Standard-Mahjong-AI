"""打印 learner checkpoint 的 update 数与 eval 信息(校验热启动来源)。"""
import sys
import torch

p = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
c = p.get('counters', {})
ev = p.get('eval', {}) or {}
print('updates=%s  eval.comp_mean=%s  eval.gen_mean=%s'
      % (c.get('updates'), ev.get('comp_mean'), ev.get('gen_mean')))
