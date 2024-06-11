import torch
import torch.nn as nn
from modules.casual_conv_blocks import CausalConv1d
from collections import OrderedDict

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # nf = 128
        # self.backbone = nn.Sequential(OrderedDict)
        # down = AA
        # self.backbone.add_module("downsample", down)
        self.conv = nn.Sequential(
            nn.ConstantPad1d((2, 0), 0),
            nn.Conv1d(1, 1, 3, 1, 0)
        )
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            with torch.no_grad():
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

if __name__ == "__main__":
    x = torch.arange(1, 11, 1).view(1, 1, -1).float().requires_grad_(False)    
    with torch.no_grad():
        E = Encoder()
        C = CausalConv1d(1, 1, 3, stride=1)        
        y = C(x)
        y2 = E(x)        
        print(y)
        print(y2)
        print(x.shape, y.shape, y2.shape)