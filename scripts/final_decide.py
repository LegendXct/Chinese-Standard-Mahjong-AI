# -*- coding: utf-8 -*-
"""D-day 终选决策(配合 scripts/final_select.sh)。

用法:
  python scripts/final_decide.py screen <tmpdir>
      读 screen_evalbest.md / screen_last.md,打印配对差值更高者的 ckpt 路径
  python scripts/final_decide.py report <rl_final> <sl_ckpt> <tmpdir> <out_md>
      按预注册规则出决策,写终选报告,stdout 只打印胜者 ckpt 路径
"""
import re
import sys
import time

EVAL_BEST = 'checkpoints/rl_v1/eval_best.pt'
LAST = 'checkpoints/rl_v1/last.pt'

RULE_TEXT = (
    '预注册决策规则(执行前已定,不看结果改规则):\n'
    '- 配对差值 >= +2*SE  -> 上 RL(显著更强)\n'
    '- 落在 +-2*SE 内     -> 上 RL(同强度取训练更充分者,且 RL 对 RuleBot 更强)\n'
    '- 配对差值 <= -2*SE  -> 上 SL-B,并告警深查 RL 训练\n'
)


def parse_md(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    m = re.search(r'同墙配对差值 \| (-?[\d.]+) ± ([\d.]+)', text)
    return float(m.group(1)), float(m.group(2)), text


def main():
    mode = sys.argv[1]
    if mode == 'screen':
        tmpdir = sys.argv[2]
        d_eb, _, _ = parse_md(tmpdir + '/screen_evalbest.md')
        d_la, _, _ = parse_md(tmpdir + '/screen_last.md')
        print(EVAL_BEST if d_eb >= d_la else LAST)
        return
    assert mode == 'report'
    rl_final, sl_ckpt, tmpdir, out = sys.argv[2:6]
    d, se, duel_md = parse_md(tmpdir + '/duel.md')
    d_eb, se_eb, _ = parse_md(tmpdir + '/screen_evalbest.md')
    d_la, se_la, _ = parse_md(tmpdir + '/screen_last.md')
    dr, ser, rule_md = parse_md(tmpdir + '/rule.md')

    if d >= 2 * se:
        decision, winner = 'RL 上线(显著强于 SL,d >= +2SE)', rl_final
    elif d > -2 * se:
        decision, winner = 'RL 上线(±2SE 内同强度,取训练更充分者)', rl_final
    else:
        decision, winner = ('SL-B 上线 !!!ALERT!!! RL 显著弱于 SL,需深查训练',
                            sl_ckpt)

    lines = [
        '# D-day 终选报告 (%s)' % time.strftime('%F %T'),
        '',
        '## 决策规则',
        '```',
        RULE_TEXT.rstrip(),
        '```',
        '',
        '## a) 候选快筛(各 500 组 vs sl_v1_champion)',
        '- eval_best.pt: %.3f ± %.3f' % (d_eb, se_eb),
        '- last.pt:      %.3f ± %.3f' % (d_la, se_la),
        '- rl_final = **%s**' % rl_final,
        '',
        '## b) 终选对决(2000 组复式,argmax 双方)',
        duel_md,
        '',
        '## c) 锚点核对(1000 组 vs RuleBot)',
        rule_md,
        '- 参照:sl_v1_champion vs RuleBot 为 +6.81±0.80(M2)',
        '',
        '## 决策',
        '- 终选配对差值 d = %.3f ± %.3f(2SE 阈 = %.3f)' % (d, se, 2 * se),
        '- **%s**' % decision,
        '- 上线 ckpt:`%s`' % winner,
        '',
    ]
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(winner)


if __name__ == '__main__':
    main()
