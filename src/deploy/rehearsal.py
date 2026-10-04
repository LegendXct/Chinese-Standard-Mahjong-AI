"""M4.0 协议级彩排:完全模拟 Botzone 交互驱动 bot 子进程。

- 裁判:env_fixed;bot 坐 0 号位,经 stdin/stdout JSON 协议(含长时运行控制行)
  由本驱动按 Botzone 判官语义喂请求(CHI/PENG 与后续弃牌合并为一条等);
- 其余 3 座:本进程 NNPolicy(argmax);
- 一致性对拍:bot 的每个决策与本进程参考模型(加载同一 fp16 权重)argmax 比对;
- 统计:每回合延迟 p50/p95/p99、常驻/加载 maxrss、非法/崩溃/协议错位计数。

用法(主环境,bot 子进程用 botzone-compat 解释器):
  python src/deploy/rehearsal.py --weights tmp/deploy/sl_v1_B.fp16.pt --games 100 \
      --bot-python /root/miniconda3/envs/botzone-compat/bin/python
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _SRC)
from feature_v1 import FeatureAgentV1  # noqa: E402
from env_fixed import MahjongGBEnvFixed  # noqa: E402
from model_v1 import MahjongNetV1, InputDictAdapter  # noqa: E402
from model_v2 import MahjongNetV2, InputDictAdapterV2  # noqa: E402
from arena.policies import NNPolicy, NNPolicyV2  # noqa: E402
from arena.duplicate import make_wall  # noqa: E402

B = 0  # bot 座位


class BotProc(object):

    def __init__(self, bot_python, weights, infer_mode='fp16_resident',
                 bot_main='deploy/bot_main.py'):
        # 干净最小环境:不让主环境的 LD_LIBRARY_PATH/PYTHONPATH 污染 bot 的
        # torch 1.10(实测污染会使 RSS 241MB -> 647MB),也更贴近 Botzone
        env = {'PATH': '/usr/bin:/bin', 'HOME': os.environ.get('HOME', '/root'),
               'LANG': os.environ.get('LANG', 'C.UTF-8'), 'BOT_WEIGHTS': weights,
               'BOT_INFER_MODE': infer_mode}
        self.t_spawn = time.perf_counter()
        self.p = subprocess.Popen(
            [bot_python, os.path.join(_SRC, bot_main)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, universal_newlines=True, bufsize=1)
        self.first = True
        self.first_turn_total_s = None   # 进程启动+import+权重加载+fp32转换+首决策
        self.debug1 = None
        self.turn_ms = []
        self.maxrss = []

    def ask(self, req):
        if self.first:
            payload = json.dumps({'requests': [req], 'responses': []})
            self.first = False
        else:
            payload = json.dumps(req)
        t0 = time.perf_counter()
        self.p.stdin.write(payload + '\n')
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        keep = self.p.stdout.readline()
        dt = (time.perf_counter() - t0) * 1000
        if not line or '>>>BOTZONE_REQUEST_KEEP_RUNNING<<<' not in keep:
            raise RuntimeError('protocol desync: %r / %r' % (line, keep))
        out = json.loads(line)
        dbg = json.loads(out.get('debug', '{}'))
        self.turn_ms.append(dbg.get('turn_wall_ms', dt))
        self.maxrss.append(dbg.get('maxrss_kb', -1))
        if self.debug1 is None:
            self.debug1 = dbg
            self.first_turn_total_s = round(time.perf_counter() - self.t_spawn, 3)
        if 'err' in dbg:
            raise RuntimeError('bot reported err: %s' % dbg['err'])
        return out['response'].strip()

    def close(self):
        try:
            self.p.terminate()
        except Exception:
            pass


def bot_resp_to_action(agent, resp, cur_tile, state):
    """bot 的 botzone 响应 -> (env 动作 id, pending_discard)。"""
    t = resp.split()
    if state == 1:   # 自己摸牌后
        if t[0] == 'HU':
            return agent.response2action('Hu'), None
        if t[0] == 'PLAY':
            return agent.response2action('Play %s' % t[1]), None
        if t[0] == 'GANG':
            return agent.response2action('AnGang %s' % t[1]), None
        if t[0] == 'BUGANG':
            return agent.response2action('BuGang %s' % t[1]), None
    elif state == 2:  # 响应他家弃牌
        if t[0] == 'HU':
            return agent.response2action('Hu'), None
        if t[0] == 'PASS':
            return 0, None
        if t[0] == 'GANG':
            return agent.response2action('Gang %s' % cur_tile), None
        if t[0] == 'PENG':
            return agent.response2action('Peng %s' % cur_tile), t[1]
        if t[0] == 'CHI':
            return agent.response2action('Chi %s %s' % (cur_tile, t[1])), t[2]
    elif state == 3:  # 抢杠窗口
        return (agent.response2action('Hu') if t[0] == 'HU' else 0), None
    raise RuntimeError('bad bot resp %r in state %s' % (resp, state))


def play_game(bot, ref_policy, env, wall, wind, stats):
    obs = env.reset(prevalentWind=wind, tileWall=wall)
    agent = env.agents[B]
    bot.ask('0 %d %d' % (B, wind))
    bot.ask('1 0 0 0 0 %s' % ' '.join(env.hands[B]))
    pending_discard = [None]
    bot_next_action = [None]
    pending_claim_echo = [None]   # (p, 'CHI mid'/'PENG') 等待弃牌合并

    def send(req, decision_state=None, ref_obs=None):
        r = bot.ask(req)
        if decision_state is not None:
            a, pd = bot_resp_to_action(agent, r, env.curTile, decision_state)
            bot_next_action[0] = a
            pending_discard[0] = pd
            if ref_obs is not None:
                ref_a = ref_policy.act(ref_obs)
                stats['cmp_total'] += 1
                if ref_a != a:
                    stats['cmp_mismatch'] += 1
                    stats['mismatch_ex'].append((req, r, int(ref_a), int(a)))
        return r

    # 开局:player0(=bot)已摸第一张
    if env.curPlayer == B:
        send('2 %s' % env.curTile, 1, obs['player_1'])
    else:
        send('3 %d DRAW' % env.curPlayer)

    done = False
    steps = 0
    while not done:
        steps += 1
        assert steps < 600
        state = env.state
        cur_p = env.curPlayer
        cur_tile = env.curTile
        action_dict = {}
        for name, o in obs.items():
            seat = int(name.split('_')[1]) - 1
            if seat == B:
                if state == 0:
                    action_dict[name] = agent.response2action('Play %s' % pending_discard[0])
                    # 对拍:状态0的弃牌决策(bot 在两段式中已预决策)
                    ref_a = ref_policy.act(o)
                    stats['cmp_total'] += 1
                    if ref_a != action_dict[name]:
                        stats['cmp_mismatch'] += 1
                        stats['mismatch_ex'].append(('state0', pending_discard[0],
                                                     int(ref_a), int(action_dict[name])))
                else:
                    action_dict[name] = bot_next_action[0]
            else:
                legal = np.flatnonzero(o['action_mask'])
                action_dict[name] = (int(legal[0]) if len(legal) == 1
                                     else ref_policy.act(o))
        obs, rew, done = env.step(action_dict)
        new_state = env.state

        # ---- 把这次转移翻译成发给 bot 的请求(Botzone 判官语义)----
        if state in (0, 1):
            resp = env.agents[cur_p].action2response(action_dict['player_%d' % (cur_p + 1)]).split()
            if state == 1 and resp[0] == 'Gang':
                send('3 %d GANG' % cur_p)
                if not done and env.curPlayer == cur_p and new_state == 1:
                    if cur_p == B:
                        send('2 %s' % env.curTile, 1, obs.get('player_%d' % (B + 1)))
                    else:
                        send('3 %d DRAW' % cur_p)
            elif state == 1 and resp[0] == 'BuGang':
                send('3 %d BUGANG %s' % (cur_p, resp[1]),
                     3 if cur_p != B else None,
                     obs.get('player_%d' % (B + 1)) if cur_p != B else None)
            elif resp[0] == 'Play':
                if pending_claim_echo[0]:
                    p0, head = pending_claim_echo[0]
                    pending_claim_echo[0] = None
                    send('3 %d %s %s' % (p0, head, resp[1]),
                         2 if p0 != B and not done else None,
                         obs.get('player_%d' % (B + 1)) if p0 != B else None)
                else:
                    send('3 %d PLAY %s' % (cur_p, resp[1]),
                         2 if cur_p != B and not done else None,
                         obs.get('player_%d' % (B + 1)) if cur_p != B else None)
                if done:
                    pass
            elif resp[0] == 'Hu':
                pass   # 终局,无后续请求
        elif state == 2:
            # 看 env 实际执行了谁的宣告
            execd = None
            for j in range(1, 4):
                q = (cur_p + j) % 4
                r = env.agents[q].action2response(action_dict['player_%d' % (q + 1)]).split()
                if r[0] == 'Hu':
                    execd = ('HU', q, r)
                    break
            if execd is None:
                for j in range(1, 4):
                    q = (cur_p + j) % 4
                    r = env.agents[q].action2response(action_dict['player_%d' % (q + 1)]).split()
                    if r[0] in ('Gang', 'Peng'):
                        execd = (r[0].upper(), q, r)
                        break
                else:
                    q = (cur_p + 1) % 4
                    r = env.agents[q].action2response(action_dict['player_%d' % (q + 1)]).split()
                    if r[0] == 'Chi':
                        execd = ('CHI', q, r)
            if execd is None:
                if not done:   # 全过,下家摸牌
                    if env.curPlayer == B:
                        send('2 %s' % env.curTile, 1, obs.get('player_%d' % (B + 1)))
                    else:
                        send('3 %d DRAW' % env.curPlayer)
            elif execd[0] == 'HU':
                pass
            elif execd[0] == 'GANG':
                send('3 %d GANG' % execd[1])
                if not done:
                    if env.curPlayer == B:
                        send('2 %s' % env.curTile, 1, obs.get('player_%d' % (B + 1)))
                    else:
                        send('3 %d DRAW' % execd[1])
            elif execd[0] == 'PENG':
                pending_claim_echo[0] = (execd[1], 'PENG')
            elif execd[0] == 'CHI':
                pending_claim_echo[0] = (execd[1], 'CHI %s' % execd[2][1])
        elif state == 3:
            hu_q = None
            for j in range(1, 4):
                q = (cur_p + j) % 4
                if env.agents[q].action2response(
                        action_dict['player_%d' % (q + 1)]) == 'Hu':
                    hu_q = q
                    break
            if hu_q is None and not done:
                if env.curPlayer == B:
                    send('2 %s' % env.curTile, 1, obs.get('player_%d' % (B + 1)))
                else:
                    send('3 %d DRAW' % cur_p)

    scores = [rew['player_%d' % (i + 1)] for i in range(4)]
    if max(scores) == 10:
        stats['invalid'] += 1
    stats['games'] += 1
    stats['bot_score'] += scores[B]
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='fp16 权重(bot 与参考模型同源)')
    ap.add_argument('--games', type=int, default=100)
    ap.add_argument('--bot-python', default='/root/miniconda3/envs/botzone-compat/bin/python')
    ap.add_argument('--tag', default='B')
    ap.add_argument('--infer-mode', default='fp16_resident',
                    choices=['fp32', 'fp16_resident'])
    ap.add_argument('--model-version', default='v1', choices=['v1', 'v2'])
    args = ap.parse_args()
    bot_main = 'deploy/bot_main_v2.py' if args.model_version == 'v2' else 'deploy/bot_main.py'

    # 先 fork bot 再加载参考模型:子进程 ru_maxrss 会把 fork 时父进程的 COW 页
    # 计入峰值(实测把 247MB 的 bot 测成 647MB),故让 fork 发生在驱动器还小时
    bot = BotProc(args.bot_python, args.weights, args.infer_mode, bot_main)
    if args.model_version == 'v2':
        net, _ = MahjongNetV2.load(args.weights)   # fp16->fp32,与 bot 同源
        ref = NNPolicyV2(InputDictAdapterV2(net))
    else:
        net, _ = MahjongNetV1.load(args.weights)   # load 内部 fp16->fp32,与 bot 同源
        ref = NNPolicy(InputDictAdapter(net))
    env = MahjongGBEnvFixed(config={'agent_clz': FeatureAgentV1})
    stats = {'games': 0, 'invalid': 0, 'crash': 0, 'cmp_total': 0, 'cmp_mismatch': 0,
             'bot_score': 0, 'mismatch_ex': []}
    t0 = time.time()
    all_ms, all_rss = [], []
    debug1 = None
    first_turn_total = bot.first_turn_total_s  # 由首局首请求填充
    for g in range(args.games):
        try:
            play_game(bot, ref, env, make_wall(('rehearsal', g)), g % 4, stats)
        except Exception as e:
            stats['crash'] += 1
            stats.setdefault('crash_ex', []).append('%d: %r' % (g, e))
            bot.close()
            all_ms += bot.turn_ms
            all_rss += bot.maxrss
            debug1 = debug1 or bot.debug1
            bot = BotProc(args.bot_python, args.weights, args.infer_mode, bot_main)
        if (g + 1) % 20 == 0:
            print('  game %d/%d (cmp %d/%d)' % (g + 1, args.games,
                  stats['cmp_total'] - stats['cmp_mismatch'], stats['cmp_total']),
                  file=sys.stderr, flush=True)
    all_ms += bot.turn_ms
    all_rss += bot.maxrss
    debug1 = debug1 or bot.debug1
    first_turn_total = first_turn_total or bot.first_turn_total_s
    ms = np.array([m for m in all_ms if isinstance(m, (int, float))])
    rss = [m for m in all_rss if isinstance(m, int) and m > 0]
    print(json.dumps({
        'tag': args.tag, 'infer_mode': args.infer_mode,
        'games': stats['games'], 'invalid': stats['invalid'],
        'crash': stats['crash'],
        'consistency': '%d/%d' % (stats['cmp_total'] - stats['cmp_mismatch'],
                                  stats['cmp_total']),
        'mismatch_examples': stats['mismatch_ex'][:5],
        'crash_examples': stats.get('crash_ex', [])[:3],
        'turn_ms': {'p50': float(np.percentile(ms, 50)), 'p95': float(np.percentile(ms, 95)),
                    'p99': float(np.percentile(ms, 99))} if len(ms) else None,
        'maxrss_mb': {'resident': round(max(rss) / 1024, 1) if rss else None,
                      'load': round(debug1.get('rss_after_load', -1) / 1024, 1)
                      if debug1 else None},
        'first_turn_total_s': first_turn_total,
        'debug1': debug1, 'elapsed_s': round(time.time() - t0, 1),
        'avg_bot_score': round(stats['bot_score'] / max(stats['games'], 1), 2),
    }, ensure_ascii=False, indent=1))
    bot.close()


if __name__ == '__main__':
    main()
