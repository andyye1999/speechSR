import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

def create_net(args):
    net = Net()
    return net