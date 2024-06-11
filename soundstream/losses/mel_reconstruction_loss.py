import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from soundstream.modules.stft import Audio2Mel
import numpy as np

class SpectralReconstructionLoss(_Loss):
    def __init__(self, sr, eps=1e-5, reduction='mean', device=None):
        super().__init__(reduction=reduction)
        self.sr = sr
        self.eps = eps
        self.melspec = []
        for i in range(6,11):   # 6 12
            s = 2**i            
            self.melspec.append(Audio2Mel(n_fft=s, win_length=s, hop_length=s//4, n_mel_channels=64, sampling_rate=sr).to(device))
        

    def forward(self, G_x, x):
        L = 0
        rng = np.arange(6, 11, 1)   # 6 12 1
        for idx, i in enumerate(rng):
            s = 2**i
            alpha_s = (s/2)**0.5
            S_x = self.melspec[idx](x)
            S_G_x = self.melspec[idx](G_x)
            l1 = F.l1_loss(S_G_x, S_x, reduction=self.reduction)
            l2 = F.mse_loss(S_G_x, S_x, reduction=self.reduction)
            L += l1 + alpha_s*l2            
        return L