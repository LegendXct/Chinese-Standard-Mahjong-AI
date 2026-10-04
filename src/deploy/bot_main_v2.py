# -*- coding: utf-8 -*-
"""Botzone 正式 bot(v2 双输入):MahjongNetV2 + FeatureAgentV1 状态机 + vec 现算。

- 与 bot_main.py(v1)完全平行、互不影响;**独立 bot 槽位**,绝不替换 Round 1 的 rl_v1_B。
- vec 由 feature_v2.compute_vec_from_obs(或 117 维 feature_v2_ext)**从 obs 现算**
  (与训练/arena 同一真源),按权重 config['vec_dim'] 自动择函数。
- 权重:/data/ 下 fp16 state_dict;fp16_resident 逐算子转 fp32 控内存。
- 决策/两段式吃碰/防御回退/协议层与 bot_main.py 一致。严格 Python 3.6 / torch 1.10。
"""
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('SHANTEN_CACHE_L1', '65536')
os.environ.setdefault('SHANTEN_CACHE_L2', '65536')
os.environ.setdefault('SHANTEN_CACHE_WAIT', '16384')
os.environ.setdefault('SHANTEN_CACHE_DISCARD', '16384')

START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WEIGHTS = os.environ.get('BOT_WEIGHTS', '/data/sl_v2_best.fp16.pt')  # make_bundle 改写
INFER_MODE = os.environ.get('BOT_INFER_MODE', 'fp16_resident')
MODEL_TAG = '%s|%s|v2' % (os.path.basename(WEIGHTS), INFER_MODE)

DEBUG1 = {}
VECFN = None


def rss_kb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return -1


def load_all():
    """import + 模型加载 + 按 vec_dim 选 vec 函数,失败信息进 DEBUG1(决策侧回退)。"""
    global VECFN
    d = DEBUG1
    d['py'] = sys.version.split()[0]
    model = None
    agent_cls = None
    try:
        t0 = time.time()
        import torch
        torch.set_num_threads(1)
        d['torch'] = torch.__version__
        d['torch_import_s'] = round(time.time() - t0, 3)
    except Exception as e:
        d['torch'] = 'ERR %r' % (e,)
        return None, None
    try:
        from MahjongGB import MahjongFanCalculator  # noqa
        d['mahjonggb'] = 'ok'
    except Exception as e:
        d['mahjonggb'] = 'ERR %r' % (e,)
    try:
        from feature_v1 import FeatureAgentV1
        agent_cls = FeatureAgentV1
    except Exception as e:
        d['feature'] = 'ERR %r' % (e,)
        return None, None
    try:
        from model_v2 import MahjongNetV2, FP16ResidentNetV2
        t0 = time.time()
        payload = torch.load(WEIGHTS, map_location='cpu')
        vec_dim = int(payload['config'].get('vec_dim', 107))
        d['vec_dim'] = vec_dim
        if vec_dim >= 117:
            from feature_v2_ext import compute_vec_ext_from_obs as vfn
        else:
            from feature_v2 import compute_vec_from_obs as vfn
        VECFN = vfn
        if INFER_MODE == 'fp16_resident':
            model = FP16ResidentNetV2(payload['state_dict'], payload['config'])
            del payload
        else:
            model = MahjongNetV2.from_config(payload['config'])
            own = model.state_dict()
            src = payload['state_dict']
            for k in list(src.keys()):
                own[k].copy_(src[k].float())
                del src[k]
            del payload
            model.eval()
        d['load_s'] = round(time.time() - t0, 3)
        d['rss_after_load'] = rss_kb()
    except Exception as e:
        d['model'] = 'ERR %r' % (e,)
        model = None
    return model, agent_cls


MODEL, AGENT_CLS = load_all()

try:
    import torch
    import numpy as np
    from feature_v1 import SCALED_CHANNELS
    _SCALE = torch.ones(102, 1, 1)
    for _c in SCALED_CHANNELS:
        _SCALE[_c] = 1.0 / 255
except Exception as _e:
    DEBUG1['scale'] = 'ERR %r' % (_e,)


def argmax_action(obs):
    raw = np.ascontiguousarray(obs['observation'])
    o = torch.from_numpy(raw).float().unsqueeze(0) * _SCALE
    vec = torch.from_numpy(np.ascontiguousarray(VECFN(raw))).float().unsqueeze(0)
    m = torch.from_numpy(np.ascontiguousarray(obs['action_mask'])).float().unsqueeze(0)
    with torch.no_grad():
        logits, _ = MODEL(o, vec)
        logits = logits + (1.0 - m) * (-1e9)
    return int(logits.argmax(dim=1).item())


class Bot(object):

    def __init__(self):
        self.agent = None
        self.seat = -1
        self.zimo = False
        self.angang = None
        self.last_drawn = None

    def decide(self, obs):
        return argmax_action(obs)

    def handle(self, req):
        t = req.split()
        if t[0] == '0':
            self.seat = int(t[1])
            self.agent = AGENT_CLS(self.seat)
            self.agent.request2obs('Wind %s' % t[2])
            return 'PASS'
        if t[0] == '1':
            self.agent.request2obs(' '.join(['Deal'] + t[5:]))
            return 'PASS'
        if t[0] == '2':
            self.last_drawn = t[1]
            obs = self.agent.request2obs('Draw %s' % t[1])
            resp = self.agent.action2response(self.decide(obs)).split()
            if resp[0] == 'Hu':
                return 'HU'
            if resp[0] == 'Play':
                return 'PLAY %s' % resp[1]
            if resp[0] == 'Gang':
                self.angang = resp[1]
                return 'GANG %s' % resp[1]
            if resp[0] == 'BuGang':
                return 'BUGANG %s' % resp[1]
            return 'PLAY %s' % t[1]
        p = int(t[1])
        ev = t[2]
        if ev == 'DRAW':
            self.agent.request2obs('Player %d Draw' % p)
            self.zimo = True
            return 'PASS'
        if ev == 'GANG':
            if p == self.seat and self.angang:
                self.agent.request2obs('Player %d AnGang %s' % (p, self.angang))
            elif self.zimo:
                self.agent.request2obs('Player %d AnGang' % p)
            else:
                self.agent.request2obs('Player %d Gang' % p)
            return 'PASS'
        if ev == 'BUGANG':
            obs = self.agent.request2obs('Player %d BuGang %s' % (p, t[3]))
            if p == self.seat:
                return 'PASS'
            resp = self.agent.action2response(self.decide(obs))
            return 'HU' if resp == 'Hu' else 'PASS'
        self.zimo = False
        if ev == 'CHI':
            self.agent.request2obs('Player %d Chi %s' % (p, t[3]))
        elif ev == 'PENG':
            self.agent.request2obs('Player %d Peng' % p)
        obs = self.agent.request2obs('Player %d Play %s' % (p, t[-1]))
        if p == self.seat:
            return 'PASS'
        resp = self.agent.action2response(self.decide(obs)).split()
        if resp[0] == 'Hu':
            return 'HU'
        if resp[0] == 'Pass':
            return 'PASS'
        if resp[0] == 'Gang':
            self.angang = None
            return 'GANG'
        if resp[0] in ('Peng', 'Chi'):
            obs2 = self.agent.request2obs('Player %d %s' % (self.seat, ' '.join(resp)))
            resp2 = self.agent.action2response(self.decide(obs2)).split()
            self.agent.request2obs('Player %d Un%s' % (self.seat, ' '.join(resp)))
            return ' '.join([resp[0].upper()] + resp[1:] + [resp2[-1]])
        return 'PASS'

    def fallback(self, req):
        t = req.split()
        if t and t[0] == '2':
            return 'PLAY %s' % t[1]
        return 'PASS'


BOT = Bot()
TURN = [0]


def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        t0 = time.time()
        TURN[0] += 1
        try:
            obj = json.loads(line)
        except Exception:
            obj = line
        if isinstance(obj, dict) and 'requests' in obj:
            reqs = [str(x) for x in obj['requests']]
        else:
            reqs = [str(obj)]
        response = 'PASS'
        err = None
        for i, req in enumerate(reqs):
            is_last = (i == len(reqs) - 1)
            try:
                if MODEL is None or AGENT_CLS is None:
                    raise RuntimeError('model/agent not loaded')
                r = BOT.handle(req)
                if is_last:
                    response = r
            except Exception as e:
                err = '%r' % (e,)
                if is_last:
                    response = BOT.fallback(req)
        debug = {'turn': TURN[0], 'turn_wall_ms': round((time.time() - t0) * 1000, 1),
                 'maxrss_kb': rss_kb(), 'model_tag': MODEL_TAG}
        if TURN[0] == 1:
            debug.update(DEBUG1)
        if err:
            debug['err'] = err
        print(json.dumps({'response': response, 'debug': json.dumps(debug)}))
        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
