import torch
import torch.nn as nn
import torch.nn.functional as F

class AADownsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(AADownsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        ha = torch.arange(1, filt_size//2+1+1, 1)
        a = torch.cat((ha, ha.flip(dims=[-1,])[1:])).float()
        a = a / a.sum()
        filt = a[None, :]
        self.register_buffer('filt', filt[None, :, :].repeat((self.channels, 1, 1)))

    def forward(self, x):
        x_pad = F.pad(x, (self.filt_size//2, self.filt_size//2), "reflect")
        y = F.conv1d(x_pad, self.filt, stride=self.stride, padding=0, groups=x.shape[1])
        return y


class Down(nn.Module):
    def __init__(self, channels, d=2, k=3):
        super().__init__()
        kk = d + 1
        self.down = nn.Sequential(
            nn.ReflectionPad1d(kk // 2),
            nn.Conv1d(channels, channels*2, kernel_size=kk, stride=1, bias=False),
            nn.BatchNorm1d(channels*2),
            nn.LeakyReLU(0.2, True),
            AADownsample(channels=channels*2, stride=d, filt_size=k)
        )

    def forward(self, x):
        x = self.down(x)
        return x