"""MahjongNetV2Dual:忠实复刻 Arthals SelfVecModel 的双输入结构,适配我们的维度 + 价值头。

Arthals 原版(zhuozhiyongde Fundamentals-of-AI-2024Spring-PKU/Homework/10-Mahjong/model.py
SelfVecModel)关键点:
- **2-conv stem**:Conv(obs,128)→BN→GELU→Conv(128,hidden)→BN→GELU;
- **后激活残差块** ×20:Conv→BN→GELU→Conv→BN,然后 `x = GELU(x + block(x))`(非预激活);
- **融合**:把整张空间特征图展平(hidden×4×9)再 **拼接原始 vec**,过一个 **三层下采样 MLP**
  (维度由 `((flat+vec)/235)^(1/3)` 立方根比例自动推出,Arthals: 4725→1736→632→235);
- 激活全用 GELU;归一化用 BN;**无价值头**(纯 SL 策略克隆);hidden=128、vec=117。

本复刻与 Arthals 的差异(均为我们项目既定结论/需要):
- **norm 用 GN**(我们已验证 BN 比 GN 差 2.56pp;本实验只测"结构+GELU",norm 固定 GN);
- **加价值头**(供阶段二 RL 热启动 critic;Arthals 无);
- 维度用我们的:obs 102ch、width 192(=M3′ 容量,干净隔离"结构"变量)、vec 107。
- act 可配(relu/gelu),供 M6(relu)/M7(gelu)消融。

注:width=192 下融合 MLP 很重(p_fc1 ≈ flat×d1),参数量约 33M(Arthals width=128 时约 15.5M)。
部署 fp16_resident 变体的可行性见报告评估。
"""
import torch
import torch.nn.functional as F
from torch import nn

from model_v1 import _make_norm
from feature_v2 import VEC_DIM


def _act_fn(act):
    return F.gelu if act == 'gelu' else F.relu


class DualResBlock(nn.Module):
    """Arthals 后激活残差块:Conv-Norm-Act-Conv-Norm,x = Act(x + y)。"""

    def __init__(self, ch, norm, act):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.n1 = _make_norm(norm, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.n2 = _make_norm(norm, ch)
        self.act = _act_fn(act)

    def forward(self, x):
        y = self.act(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return self.act(x + y)


class MahjongNetV2Dual(nn.Module):

    def __init__(self, in_channels=102, width=192, depth=20, norm='gn',
                 act='relu', actions=235, vec_dim=VEC_DIM):
        super().__init__()
        self.config = {'in_channels': in_channels, 'width': width, 'depth': depth,
                       'norm': norm, 'act': act, 'actions': actions,
                       'vec_dim': vec_dim, 'arch': 'dual'}
        self.act = _act_fn(act)
        # 2-conv stem(Arthals)
        self.stem_c1 = nn.Conv2d(in_channels, 128, 3, 1, 1, bias=False)
        self.stem_n1 = _make_norm(norm, 128)
        self.stem_c2 = nn.Conv2d(128, width, 3, 1, 1, bias=False)
        self.stem_n2 = _make_norm(norm, width)
        self.blocks = nn.ModuleList([DualResBlock(width, norm, act) for _ in range(depth)])

        flat = width * 4 * 9
        merged = flat + vec_dim
        # Arthals 立方根下采样比例
        ratio = (merged / actions) ** (1.0 / 3.0)
        d1 = max(int(merged / ratio) // 8 * 8, actions)
        d2 = max(int(d1 / ratio) // 8 * 8, actions)
        self.dims = (merged, d1, d2)
        # 策略头:展平全谱 + 拼 vec -> 三层下采样 MLP
        self.p_fc1 = nn.Linear(merged, d1)
        self.p_fc2 = nn.Linear(d1, d2)
        self.p_fc3 = nn.Linear(d2, actions)
        # 价值头(轻量,off 同一展平特征)
        self.v_fc1 = nn.Linear(merged, 256)
        self.v_fc2 = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs, vec=None):
        if isinstance(obs, dict):
            vec = obs['vec']
            obs = obs['observation']
        x = self.act(self.stem_n1(self.stem_c1(obs)))
        x = self.act(self.stem_n2(self.stem_c2(x)))
        for blk in self.blocks:
            x = blk(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, vec], dim=1)
        p = self.act(self.p_fc1(x))
        p = self.act(self.p_fc2(p))
        logits = self.p_fc3(p)
        v = self.act(self.v_fc1(x))
        v = self.v_fc2(v).squeeze(-1)
        return logits, v

    @classmethod
    def from_config(cls, cfg):
        cfg = {k: v for k, v in cfg.items() if k != 'arch'}
        return cls(**cfg)

    def save(self, path, half=False, extra=None):
        sd = self.state_dict()
        if half:
            sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
        payload = {'config': self.config, 'state_dict': sd}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path, map_location='cpu'):
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        model = cls.from_config(payload['config'])
        sd = {k: (v.float() if v.is_floating_point() else v)
              for k, v in payload['state_dict'].items()}
        model.load_state_dict(sd)
        return model, payload

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
