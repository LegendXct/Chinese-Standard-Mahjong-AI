"""打包 Botzone 部署物:

dist/botzone_bot.zip       bot 代码包(__main__.py = bot_main + 依赖模块)
dist/sl_v1_<TAG>.fp16.pt   权重(传用户存储空间,线上路径 /data/)

用法:python scripts/make_bundle.py --ckpt checkpoints/sl_B_gn/best.pt --tag B
"""
import argparse
import os
import shutil
import sys
import zipfile

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

BUNDLE_MODULES = ['feature_v1.py', 'shanten_py.py', 'shanten_table.py', 'model_v1.py']
BUNDLE_MODULES_V2 = BUNDLE_MODULES + ['feature_v2.py', 'feature_v2_ext.py', 'model_v2.py']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arch', default='v1', choices=['v1', 'v2'],
                    help='v2=双输入 MahjongNetV2(独立 bot,用 bot_main_v2,默认名 sl_v2_best)')
    ap.add_argument('--tag', default='B')
    ap.add_argument('--name', default=None,
                    help='权重文件名(不含扩展名),默认 sl_v1_<TAG>;RL 换装时'
                         '用 --name rl_v1_B 等,zip 内默认路径随之改写')
    ap.add_argument('--infer-mode', default='fp16_resident',
                    choices=['fp32', 'fp16_resident'])
    ap.add_argument('--out', default=os.path.join(ROOT, 'dist'))
    args = ap.parse_args()
    is_v2 = (args.arch == 'v2')
    name = args.name or ('sl_v2_best' if is_v2 else 'sl_v1_%s' % args.tag)

    os.makedirs(args.out, exist_ok=True)
    # 1) 权重 fp16
    if is_v2:
        from model_v2 import MahjongNetV2
        m, p = MahjongNetV2.load(args.ckpt)
    else:
        from model_v1 import MahjongNetV1
        m, p = MahjongNetV1.load(args.ckpt)
    wpath = os.path.join(args.out, '%s.fp16.pt' % name)
    m.save(wpath, half=True, extra={'epoch': p.get('epoch'),
                                    'val_acc': p.get('val_acc'), 'tag': args.tag})
    # 2) 代码 zip(__main__.py 为入口);线上无法设环境变量,默认权重路径/推理模式改写进 __main__.py
    main_src = 'bot_main_v2.py' if is_v2 else 'bot_main.py'
    default_w = '/data/sl_v2_best.fp16.pt' if is_v2 else '/data/sl_v1_A.fp16.pt'
    modules = BUNDLE_MODULES_V2 if is_v2 else BUNDLE_MODULES
    with open(os.path.join(ROOT, 'src', 'deploy', main_src), encoding='utf-8') as f:
        main_text = f.read()
    pat_w = "os.environ.get('BOT_WEIGHTS', '%s')" % default_w
    pat_m = "os.environ.get('BOT_INFER_MODE', 'fp16_resident')"
    assert pat_w in main_text and pat_m in main_text, 'bot_main 默认值行变了,需同步'
    main_text = main_text.replace(
        pat_w, "os.environ.get('BOT_WEIGHTS', '/data/%s.fp16.pt')" % name)
    main_text = main_text.replace(
        pat_m, "os.environ.get('BOT_INFER_MODE', '%s')" % args.infer_mode)
    zpath = os.path.join(args.out, 'botzone_bot_%s.zip' % name)
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('__main__.py', main_text)
        for mod in modules:
            z.write(os.path.join(ROOT, 'src', mod), mod)
    print('bundle:')
    print('  %s (%.2f MB)' % (zpath, os.path.getsize(zpath) / 2**20))
    print('  %s (%.2f MB)' % (wpath, os.path.getsize(wpath) / 2**20))


if __name__ == '__main__':
    main()
