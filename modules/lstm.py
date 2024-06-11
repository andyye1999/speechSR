'''
Author: yehongcen && as_yhc@163.com
Date: 2024-04-23 15:36:50
LastEditors: redust as_yhc@163.com
LastEditTime: 2024-06-07 21:19:21
FilePath: \学位论文代码speechSR\modules\lstm.py
Description: 因果LSTM
Copyright (c) 2024 by yehongcen, All Rights Reserved. 
'''
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM layers module."""

from torch import nn


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
