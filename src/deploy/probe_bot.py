# -*- coding: utf-8 -*-
"""Botzone 平台探针 bot(国标麻将,单文件,不依赖任何模型)。

目的:探测平台真实环境(Python/torch/numpy 版本、import torch 耗时、
内存上限表现、每回合时延),不求棋力。

协议:Botzone JSON 交互 + 长时运行模式。
行为:自己摸牌回合打出刚摸到的那张牌(恒合法);其余请求一律 PASS。
探测信息写入每回合 response 的 debug 字段;全部 try/except 包裹,
探测失败绝不影响出牌输出。

严格 Python 3.6 语法(平台 Python 3.6 档)。
"""
import json
import sys
import time

START_TS = time.time()
TURN = [0]
PROBE_CACHE = {}


def probe_static():
    """turn1:环境一次性探测。"""
    if PROBE_CACHE:
        return PROBE_CACHE
    d = PROBE_CACHE
    try:
        d['py'] = sys.version.replace('\n', ' ')
    except Exception as e:  # noqa
        d['py'] = 'ERR %r' % (e,)
    try:
        import numpy
        d['numpy'] = numpy.__version__
    except Exception as e:
        d['numpy'] = 'ERR %r' % (e,)
    try:
        t0 = time.time()
        import torch
        d['torch'] = torch.__version__
        d['torch_import_s'] = round(time.time() - t0, 3)
    except Exception as e:
        d['torch'] = 'ERR %r' % (e,)
    d['maxrss_kb_after_imports'] = probe_rss()
    return d


def probe_rss():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception as e:
        return 'ERR %r' % (e,)


def decide(request):
    """request: token 化的请求字符串。返回 response 字符串。"""
    t = request.split()
    if not t:
        return 'PASS'
    if t[0] == '2':
        return 'PLAY %s' % t[1]   # 打出刚摸的牌,恒合法
    return 'PASS'


def handle_line(line):
    t_turn0 = time.time()
    TURN[0] += 1
    obj = None
    try:
        obj = json.loads(line)
    except Exception:
        obj = line
    if isinstance(obj, dict) and 'requests' in obj:
        request = obj['requests'][-1]
    else:
        request = obj
    if isinstance(request, dict):
        request = str(request)
    response = 'PASS'
    try:
        response = decide(str(request))
    except Exception:
        response = 'PASS'
    debug = {}
    try:
        debug['turn'] = TURN[0]
        if TURN[0] == 1:
            debug.update(probe_static())
        debug['turn_wall_ms'] = round((time.time() - t_turn0) * 1000, 2)
        debug['uptime_s'] = round(time.time() - START_TS, 2)
        debug['maxrss_kb'] = probe_rss()
    except Exception as e:
        debug = {'probe_err': '%r' % (e,)}
    print(json.dumps({'response': response, 'debug': json.dumps(debug)}))
    print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
    sys.stdout.flush()


def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        handle_line(line)


if __name__ == '__main__':
    main()
