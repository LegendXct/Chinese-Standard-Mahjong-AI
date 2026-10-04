import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
for p in (os.path.join(ROOT, 'src'), os.path.join(ROOT, 'baseline', 'RL')):
    if p not in sys.path:
        sys.path.insert(0, p)

SAMPLE_TXT = os.path.join(ROOT, 'baseline', 'SL', 'data', 'sample.txt')
DATA_TXT = os.path.join(ROOT, 'baseline', 'SL', 'data', 'data.txt')


def load_baseline_sl_feature():
    """以独立模块名加载 baseline/SL/feature.py(避免与 RL/feature.py 同名冲突)。"""
    import importlib.util
    path = os.path.join(ROOT, 'baseline', 'SL', 'feature.py')
    spec = importlib.util.spec_from_file_location('baseline_sl_feature', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
