"""MahjongNetV1:预激活 ResNet 主干 + 策略/价值双头。

- 输入 (B, 102, 4, 9) float(已 normalize_obs 还原),空间全程 4×9 不降采样
- Norm 默认 GroupNorm(8)(M3 PPO 的小相关 minibatch 下 BN running stats 漂移
  是经典不稳定源,GN 与 batch 无关);BN 仅作探针对比项
- forward(obs_float) -> (logits 原始未掩码, value);mask 在 loss/推理侧处理
- save/load 把 config 与权重同存,M3 与 arena NNPolicy 直接复用
"""
import torch
import torch.nn.functional as F
from torch import nn

CONFIGS = {
    'A': {'width': 128, 'depth': 20, 'norm': 'gn'},
    'B': {'width': 192, 'depth': 20, 'norm': 'gn'},
}


def _make_norm(norm, ch):
    if norm == 'gn':
        return nn.GroupNorm(8, ch)
    if norm == 'bn':
        return nn.BatchNorm2d(ch)
    raise ValueError(norm)


class PreActBlock(nn.Module):
    def __init__(self, ch, norm, act='relu'):
        super().__init__()
        self.n1 = _make_norm(norm, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.n2 = _make_norm(norm, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.act = F.gelu if act == 'gelu' else F.relu

    def forward(self, x):
        y = self.c1(self.act(self.n1(x)))
        y = self.c2(self.act(self.n2(y)))
        return x + y


class MahjongNetV1(nn.Module):

    def __init__(self, in_channels=102, width=128, depth=20, norm='gn', actions=235):
        super().__init__()
        self.config = {'in_channels': in_channels, 'width': width, 'depth': depth,
                       'norm': norm, 'actions': actions}
        self.stem = nn.Conv2d(in_channels, width, 3, 1, 1, bias=False)
        self.blocks = nn.Sequential(*[PreActBlock(width, norm) for _ in range(depth)])
        self.final_norm = _make_norm(norm, width)
        # 策略头
        self.p_conv = nn.Conv2d(width, 64, 1, bias=False)
        self.p_norm = _make_norm(norm, 64)
        self.p_fc = nn.Linear(64 * 36, actions)
        # 价值头
        self.v_conv = nn.Conv2d(width, 32, 1, bias=False)
        self.v_norm = _make_norm(norm, 32)
        self.v_fc1 = nn.Linear(32 * 36, 256)
        self.v_fc2 = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for blk in self.blocks:
            nn.init.zeros_(blk.n2.weight)  # 残差分支零初始化,起步近似恒等

    def forward(self, obs):
        x = self.stem(obs)
        x = self.blocks(x)
        x = torch.relu(self.final_norm(x))
        p = torch.relu(self.p_norm(self.p_conv(x))).flatten(1)
        logits = self.p_fc(p)
        v = torch.relu(self.v_norm(self.v_conv(x))).flatten(1)
        v = self.v_fc2(torch.relu(self.v_fc1(v)))
        return logits, v.squeeze(-1)

    # ---------------- 工具 ----------------
    @classmethod
    def from_config(cls, cfg):
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
        except TypeError:   # torch<2.0(Botzone 1.10)无 weights_only 参数
            payload = torch.load(path, map_location=map_location)
        model = cls.from_config(payload['config'])
        sd = {k: (v.float() if v.is_floating_point() else v)
              for k, v in payload['state_dict'].items()}
        model.load_state_dict(sd)
        return model, payload

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class FP16ResidentNet(object):
    """fp16 state_dict 常驻、前向逐算子临时转 fp32 的推理网(Botzone 256MB 用)。

    torch 1.10 CPU 无 fp16 算子,计算必须 fp32;但整模型 .float() 意味着双份
    常驻(B:fp16 29MB + fp32 57MB)。本类不构造 nn.Module,直接对 fp16
    state_dict 做 functional 前向,每个算子只临时转换自己的权重
    (峰值 +~1.3MB,B 最大权重 192*192*3*3)。加载侧也因此省去 from_config
    构造 fp32 模型的 57MB 峰值。
    数学路径与 MahjongNetV1.forward 逐算子一致(nn.Conv2d/GroupNorm/Linear
    的 forward 即 F.conv2d/group_norm/linear,且权重同为 fp16->float() 值),
    输出应与现行 fp32 部署路径逐位一致。严格 py3.6 + torch1.10 语法。
    """

    def __init__(self, state_dict, config):
        assert config['norm'] == 'gn', 'fp16_resident 仅支持 GroupNorm'
        self.sd = {k: (v.half() if v.is_floating_point() else v)
                   for k, v in state_dict.items()}
        self.depth = config['depth']

    def _gn(self, x, p):
        return F.group_norm(x, 8, self.sd[p + '.weight'].float(),
                            self.sd[p + '.bias'].float())

    def _conv(self, x, p, pad):
        return F.conv2d(x, self.sd[p + '.weight'].float(), None, 1, pad)

    def _fc(self, x, p):
        return F.linear(x, self.sd[p + '.weight'].float(),
                        self.sd[p + '.bias'].float())

    def __call__(self, obs):
        x = self._conv(obs, 'stem', 1)
        for i in range(self.depth):
            b = 'blocks.%d' % i
            y = self._conv(torch.relu(self._gn(x, b + '.n1')), b + '.c1', 1)
            y = self._conv(torch.relu(self._gn(y, b + '.n2')), b + '.c2', 1)
            x = x + y
        x = torch.relu(self._gn(x, 'final_norm'))
        p = torch.relu(self._gn(self._conv(x, 'p_conv', 0), 'p_norm')).flatten(1)
        logits = self._fc(p, 'p_fc')
        v = torch.relu(self._gn(self._conv(x, 'v_conv', 0), 'v_norm')).flatten(1)
        v = self._fc(torch.relu(self._fc(v, 'v_fc1')), 'v_fc2')
        return logits, v.squeeze(-1)


class InputDictAdapter(nn.Module):
    """适配 baseline/NNPolicy 的 input_dict 约定:返回 (masked_logits, value)。
    输入 observation 应已 normalize(float);mask 加 -1e9。"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, input_dict):
        logits, value = self.net(input_dict['observation'])
        mask = input_dict['action_mask']
        return logits + (1.0 - mask) * (-1e9), value
