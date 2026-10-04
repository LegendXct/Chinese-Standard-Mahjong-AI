# -*- coding: utf-8 -*-
"""Botzone 正式 bot:MahjongNetV1 + FeatureAgentV1,JSON 交互 + 长时运行。

- 权重:/data/ 下 fp16 state_dict(用户存储空间),逐参数转 fp32 控制峰值内存
- 决策:masked argmax;两段式吃碰(决定吃/碰后立刻再决策打哪张,UnChi/UnPeng 回滚)
- 防御:决策全程 try/except,异常回退恒合法动作(出牌态打刚摸的牌,响应态 PASS)
- 严格 Python 3.6 语法 / torch 1.10 API(Botzone 实测环境)
请求翻译层参照 baseline/SL/__main__.py。
"""
import json
import os
import sys
import time

# 必须在 import torch/numpy 之前:线程池按可见核数初始化会放大 RSS
# (服务器 25 核实测 647MB vs 单线程 ~150MB),Botzone 256MB 限内务必钉死
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
# 向听缓存帽:lru_cache 是内存放大器(M1 教训),单局所需极小,防长局膨胀
os.environ.setdefault('SHANTEN_CACHE_L1', '65536')
os.environ.setdefault('SHANTEN_CACHE_L2', '65536')
os.environ.setdefault('SHANTEN_CACHE_WAIT', '16384')
os.environ.setdefault('SHANTEN_CACHE_DISCARD', '16384')

START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    # 线上 bundle 平铺于 __main__.py 同目录;仓库内依赖模块在上级 src/
    if _p not in sys.path:
        sys.path.insert(0, _p)

WEIGHTS = os.environ.get('BOT_WEIGHTS', '/data/sl_v1_A.fp16.pt')  # make_bundle 按构建参数改写
INFER_MODE = os.environ.get('BOT_INFER_MODE', 'fp16_resident')
MODEL_TAG = '%s|%s' % (os.path.basename(WEIGHTS), INFER_MODE)

DEBUG1 = {}


def rss_kb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return -1


def load_all():
    """import + 模型加载,失败信息进 DEBUG1(决策侧会走回退)。"""
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
        from model_v1 import MahjongNetV1, FP16ResidentNet
        t0 = time.time()
        payload = torch.load(WEIGHTS, map_location='cpu')
        if INFER_MODE == 'fp16_resident':
            # fp16 常驻 + 前向逐算子临时转 fp32:免去 fp32 模型常驻与构造峰值
            model = FP16ResidentNet(payload['state_dict'], payload['config'])
            del payload
        else:
            model = MahjongNetV1.from_config(payload['config'])
            own = model.state_dict()
            src = payload['state_dict']
            for k in list(src.keys()):
                own[k].copy_(src[k].float())
                del src[k]                  # 逐参数转换并释放,控制峰值
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
    o = torch.from_numpy(np.ascontiguousarray(obs['observation'])).float().unsqueeze(0)
    o = o * _SCALE
    m = torch.from_numpy(np.ascontiguousarray(obs['action_mask'])).float().unsqueeze(0)
    with torch.no_grad():
        logits, _ = MODEL(o)
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
        """req: botzone 请求字符串。返回 response 字符串(botzone 格式)。"""
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
        # t[0] == '3':他家/自己的公开事件
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
        # CHI/PENG/PLAY 的最后一个 token 都是该家打出的牌
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
            # 两段式:先按已吃/碰的状态再决策打哪张,然后回滚状态等平台确认
            obs2 = self.agent.request2obs('Player %d %s' % (self.seat, ' '.join(resp)))
            resp2 = self.agent.action2response(self.decide(obs2)).split()
            self.agent.request2obs('Player %d Un%s' % (self.seat, ' '.join(resp)))
            return ' '.join([resp[0].upper()] + resp[1:] + [resp2[-1]])
        return 'PASS'

    def fallback(self, req):
        """恒合法回退:出牌态打刚摸的牌,其余 PASS。"""
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
