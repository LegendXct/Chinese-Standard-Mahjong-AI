import torch
from torch import nn

class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Flatten()
        )
        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 235)
        )
        self._value_branch = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        hidden = self._tower(obs)
        logits = self._logits(hidden)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        value_hidden = self._value_branch[0](hidden)
        value_hidden = self._value_branch[1](value_hidden)
        try:
            value = self._value_branch[2](value_hidden)
        except RuntimeError as e:
            # Work around a known CPU matmul backend issue on some aarch64 builds
            # when Linear outputs one channel.
            if value_hidden.device.type == 'cpu' and 'primitive descriptor' in str(e):
                w = self._value_branch[2].weight
                b = self._value_branch[2].bias
                value = torch.sum(value_hidden * w, dim = 1, keepdim = True)
                if b is not None:
                    value = value + b.view(1, 1)
            else:
                raise
        return masked_logits, value
