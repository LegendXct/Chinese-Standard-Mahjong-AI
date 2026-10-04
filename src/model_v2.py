"""MahjongNetV2:v1 的预激活 ResNet 主干 + 标量 vec 分支(双输入),策略/价值双头。

- 空间输入 (B,102,4,9) 过 20×width 预激活残差主干(复用 model_v1.PreActBlock);
- 主干输出经 1×1 conv 降维后展平,拼接标量 vec(feature_v2.compute_vec_from_obs,
  含每弃牌质量先验),再过 MLP 出 235 logits 与 value;
- Norm 可配 {bn,gn}:gen2 SL 阶段对标 Arthals 默认 bn(纯 SL 更准),RL 阶段可切 gn。
- forward(obs_float, vec) 或 forward(input_dict);save/load 同存 config + vec_dim。
"""
import torch
import torch.nn.functional as F
from torch import nn

from model_v1 import PreActBlock, _make_norm
from feature_v2 import VEC_DIM

CONFIGS_V2 = {
    'A': {'width': 128, 'depth': 20, 'norm': 'bn'},
    'B': {'width': 192, 'depth': 20, 'norm': 'bn'},
}


class MahjongNetV2(nn.Module):

    def __init__(self, in_channels=102, width=192, depth=20, norm='bn',
                 act='relu', actions=235, vec_dim=VEC_DIM):
        super().__init__()
        self.config = {'in_channels': in_channels, 'width': width, 'depth': depth,
                       'norm': norm, 'act': act, 'actions': actions, 'vec_dim': vec_dim}
        self.act = F.gelu if act == 'gelu' else F.relu
        self.stem = nn.Conv2d(in_channels, width, 3, 1, 1, bias=False)
        self.blocks = nn.Sequential(*[PreActBlock(width, norm, act) for _ in range(depth)])
        self.final_norm = _make_norm(norm, width)
        # 策略头:1×1 降维 -> flatten -> 拼 vec -> MLP
        self.p_conv = nn.Conv2d(width, 64, 1, bias=False)
        self.p_norm = _make_norm(norm, 64)
        self.p_fc1 = nn.Linear(64 * 36 + vec_dim, 256)
        self.p_fc2 = nn.Linear(256, actions)
        # 价值头
        self.v_conv = nn.Conv2d(width, 32, 1, bias=False)
        self.v_norm = _make_norm(norm, 32)
        self.v_fc1 = nn.Linear(32 * 36 + vec_dim, 256)
        self.v_fc2 = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for blk in self.blocks:
            nn.init.zeros_(blk.n2.weight)

    def forward(self, obs, vec=None):
        if isinstance(obs, dict):
            vec = obs['vec']
            obs = obs['observation']
        x = self.stem(obs)
        x = self.blocks(x)
        x = self.act(self.final_norm(x))
        p = self.act(self.p_norm(self.p_conv(x))).flatten(1)
        p = torch.cat([p, vec], dim=1)
        logits = self.p_fc2(self.act(self.p_fc1(p)))
        v = self.act(self.v_norm(self.v_conv(x))).flatten(1)
        v = torch.cat([v, vec], dim=1)
        v = self.v_fc2(self.act(self.v_fc1(v)))
        return logits, v.squeeze(-1)

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
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        model = cls.from_config(payload['config'])
        sd = {k: (v.float() if v.is_floating_point() else v)
              for k, v in payload['state_dict'].items()}
        model.load_state_dict(sd)
        return model, payload

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class InputDictAdapterV2(nn.Module):
    """arena/NNPolicy 约定:输入 {'observation'(已归一化),'vec','action_mask'},
    返回 (masked_logits, value)。"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, input_dict):
        logits, value = self.net(input_dict['observation'], input_dict['vec'])
        mask = input_dict['action_mask']
        return logits + (1.0 - mask) * (-1e9), value


class FP16ResidentNetV2(object):
    """model_v2 的 fp16 常驻推理变体(双输入 obs+vec),Botzone 256MB 用。

    与 model_v1.FP16ResidentNet 同思路:fp16 state_dict 常驻,前向逐算子临时 .float(),
    免去 fp32 模型常驻与构造峰值;**双输入两个头分支都覆盖**(p/v 头展平后各拼一次 vec)。
    数学路径与 MahjongNetV2.forward 逐算子一致(F.conv2d/group_norm/linear,权重同为
    fp16->float()),输出应与 fp32 部署路径逐位一致。仅支持 GroupNorm。严格 py3.6 + torch1.10。
    """

    def __init__(self, state_dict, config):
        assert config['norm'] == 'gn', 'fp16_resident 仅支持 GroupNorm'
        act = config.get('act', 'relu')
        assert act in ('relu', 'gelu'), 'unsupported act %r' % act
        self.act_fn = F.gelu if act == 'gelu' else F.relu
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

    def __call__(self, obs, vec):
        a = self.act_fn
        x = self._conv(obs, 'stem', 1)
        for i in range(self.depth):
            b = 'blocks.%d' % i
            y = self._conv(a(self._gn(x, b + '.n1')), b + '.c1', 1)
            y = self._conv(a(self._gn(y, b + '.n2')), b + '.c2', 1)
            x = x + y
        x = a(self._gn(x, 'final_norm'))
        p = a(self._gn(self._conv(x, 'p_conv', 0), 'p_norm')).flatten(1)
        p = torch.cat([p, vec], dim=1)
        logits = self._fc(a(self._fc(p, 'p_fc1')), 'p_fc2')
        v = a(self._gn(self._conv(x, 'v_conv', 0), 'v_norm')).flatten(1)
        v = torch.cat([v, vec], dim=1)
        v = self._fc(a(self._fc(v, 'v_fc1')), 'v_fc2')
        return logits, v.squeeze(-1)
