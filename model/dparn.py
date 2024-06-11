'''
Author: yehongcen && as_yhc@163.com
Date: 2024-04-23 15:36:50
LastEditors: redust as_yhc@163.com
LastEditTime: 2024-06-07 21:24:52
FilePath: \学位论文代码speechSR\model\dparn.py
Description: 第四章模型
Copyright (c) 2024 by yehongcen, All Rights Reserved. 
'''
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:32:17 2022

@author: Zhongshu.Hou & Qinwen.hu

Modules
"""
import torch
from torch import nn
import numpy as np
import math
from model.conformer import ConformerBlock
from thop import profile
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
torch.set_default_tensor_type(torch.FloatTensor)

'''
Import initialized SCM matrix
'''
Sc = np.load('D:\yhc\dparn\SpecCompress.npy').astype(np.float32) # 自适应层权重




class TSConformerBlock(nn.Module):
    def __init__(self):
        super(TSConformerBlock, self).__init__()

        self.time_conformer = ConformerBlock(dim=80,  n_head=4, ccm_kernel_size=31,
                                             ffm_dropout=0.2, attn_dropout=0.2)
        self.freq_conformer = ConformerBlock(dim=80,  n_head=4, ccm_kernel_size=31,
                                             ffm_dropout=0.2, attn_dropout=0.2)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_conformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_conformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x
'''
description: 编码器
return {*}
'''
class Encoder(nn.Module):

    def __init__(self, auto_encoder=True):
        super(Encoder, self).__init__()

        self.F = 601
        self.F_c = 256
        self.F_low = 125
        self.auto_encoder = auto_encoder

        self.flc_low = nn.Linear(self.F, self.F_low, bias=False)
        self.flc_high = nn.Linear(self.F, self.F_c - self.F_low, bias=False)

        self.conv_1 = nn.Conv2d(2, 16, kernel_size=(2, 5), stride=(1, 2), padding=(1, 1))
        self.bn_1 = nn.BatchNorm2d(16, eps=1e-8)
        self.act_1 = nn.PReLU(16)

        self.conv_2 = nn.Conv2d(16, 32, kernel_size=(2, 3), stride=(1, 1), padding=(1, 1))
        self.bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_2 = nn.PReLU(32)

        self.conv_3 = nn.Conv2d(32, 48, kernel_size=(2, 3), stride=(1, 1), padding=(1, 1))
        self.bn_3 = nn.BatchNorm2d(48, eps=1e-8)
        self.act_3 = nn.PReLU(48)

        self.conv_4 = nn.Conv2d(48, 64, kernel_size=(2, 3), stride=(1, 1), padding=(1, 1))
        self.bn_4 = nn.BatchNorm2d(64, eps=1e-8)
        self.act_4 = nn.PReLU(64)

        self.conv_5 = nn.Conv2d(64, 80, kernel_size=(1, 2), stride=(1, 1), padding=(0, 1))
        self.bn_5 = nn.BatchNorm2d(80, eps=1e-8)
        self.act_5 = nn.PReLU(80)

    def init_load(self):
        self.flc_low.weight = nn.Parameter(torch.from_numpy(Sc[:self.F_low, :]), requires_grad=False)
        self.flc_high.weight = nn.Parameter(torch.from_numpy(Sc[self.F_low:, :]), requires_grad=True)

    def forward(self, x):
        # x.shape = (Bs, F, T, 2)
        x = x.permute(0, 3, 2, 1)  # (Bs, 2, T, F)
        x = x.to(torch.float32)
        x_low = self.flc_low(x)
        x_high = self.flc_high(x)
        x = torch.cat([x_low, x_high], -1)
        x_1 = self.act_1(self.bn_1(self.conv_1(x)[:, :, :-1, :]))
        x_2 = self.act_2(self.bn_2(self.conv_2(x_1)[:, :, :-1, :]))
        x_3 = self.act_3(self.bn_3(self.conv_3(x_2)[:, :, :-1, :]))
        x_4 = self.act_4(self.bn_4(self.conv_4(x_3)[:, :, :-1, :]))
        x_5 = self.act_5(self.bn_5(self.conv_5(x_4)[:, :, :, :-1]))

        return [x_1, x_2, x_3, x_4, x_5]

'''
description: 实部解码器块
return {*}
'''
class Real_Decoder(nn.Module):
    def __init__(self, auto_encoder=True):
        super(Real_Decoder, self).__init__()
        self.F = 601
        self.F_c = 256
        self.auto_encoder = auto_encoder

        self.real_dconv_1 = nn.ConvTranspose2d(160, 64, kernel_size=(1, 2), stride=(1, 1))
        self.real_bn_1 = nn.BatchNorm2d(64, eps=1e-8)
        self.real_act_1 = nn.PReLU(64)

        self.real_dconv_2 = nn.ConvTranspose2d(128, 48, kernel_size=(2, 3), stride=(1, 1))
        self.real_bn_2 = nn.BatchNorm2d(48, eps=1e-8)
        self.real_act_2 = nn.PReLU(48)

        self.real_dconv_3 = nn.ConvTranspose2d(96, 32, kernel_size=(2, 3), stride=(1, 1))
        self.real_bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.real_act_3 = nn.PReLU(32)

        self.real_dconv_4 = nn.ConvTranspose2d(64, 16, kernel_size=(2, 3), stride=(1, 1))
        self.real_bn_4 = nn.BatchNorm2d(16, eps=1e-8)
        self.real_act_4 = nn.PReLU(16)

        self.real_dconv_5 = nn.ConvTranspose2d(32, 1, kernel_size=(2, 5), stride=(1, 2))
        self.real_bn_5 = nn.BatchNorm2d(1, eps=1e-8)
        self.real_act_5 = nn.PReLU(1)

        self.inv_flc = nn.Linear(self.F_c, self.F, bias=False)

    def forward(self, dprnn_out, encoder_out):
        skipcon_1 = torch.cat([encoder_out[4], dprnn_out], 1)
        x_1 = self.real_act_1(self.real_bn_1(self.real_dconv_1(skipcon_1)[:, :, :, :-1]))
        skipcon_2 = torch.cat([encoder_out[3], x_1], 1)
        x_2 = self.real_act_2(self.real_bn_2(self.real_dconv_2(skipcon_2)[:, :, :-1, :-2]))
        skipcon_3 = torch.cat([encoder_out[2], x_2], 1)
        x_3 = self.real_act_3(self.real_bn_3(self.real_dconv_3(skipcon_3)[:, :, :-1, :-2]))
        skipcon_4 = torch.cat([encoder_out[1], x_3], 1)
        x_4 = self.real_act_4(self.real_bn_4(self.real_dconv_4(skipcon_4)[:, :, :-1, :-2]))
        skipcon_5 = torch.cat([encoder_out[0], x_4], 1)
        x_5 = self.real_act_5(self.real_bn_5(self.real_dconv_5(skipcon_5)[:, :, :-1, :-1]))
        outp = self.inv_flc(x_5)
        return outp

'''
description: 虚部解码器块
return {*}
'''
class Imag_Decoder(nn.Module):
    def __init__(self, auto_encoder=True):
        super(Imag_Decoder, self).__init__()

        self.F = 601
        self.F_c = 256
        self.auto_encoder = auto_encoder
        self.imag_dconv_1 = nn.ConvTranspose2d(160, 64, kernel_size=(1, 2), stride=(1, 1))
        self.imag_bn_1 = nn.BatchNorm2d(64, eps=1e-8)
        self.imag_act_1 = nn.PReLU(64)

        self.imag_dconv_2 = nn.ConvTranspose2d(128, 48, kernel_size=(2, 3), stride=(1, 1))
        self.imag_bn_2 = nn.BatchNorm2d(48, eps=1e-8)
        self.imag_act_2 = nn.PReLU(48)

        self.imag_dconv_3 = nn.ConvTranspose2d(96, 32, kernel_size=(2, 3), stride=(1, 1))
        self.imag_bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_3 = nn.PReLU(32)

        self.imag_dconv_4 = nn.ConvTranspose2d(64, 16, kernel_size=(2, 3), stride=(1, 1))
        self.imag_bn_4 = nn.BatchNorm2d(16, eps=1e-8)
        self.imag_act_4 = nn.PReLU(16)

        self.imag_dconv_5 = nn.ConvTranspose2d(32, 1, kernel_size=(2, 5), stride=(1, 2))
        self.imag_bn_5 = nn.BatchNorm2d(1, eps=1e-8)
        self.imag_act_5 = nn.PReLU(1)

        self.inv_flc = nn.Linear(self.F_c, self.F, bias=False)

    def forward(self, dprnn_out, encoder_out):
        skipcon_1 = torch.cat([encoder_out[4], dprnn_out], 1)
        x_1 = self.imag_act_1(self.imag_bn_1(self.imag_dconv_1(skipcon_1)[:, :, :, :-1]))
        skipcon_2 = torch.cat([encoder_out[3], x_1], 1)
        x_2 = self.imag_act_2(self.imag_bn_2(self.imag_dconv_2(skipcon_2)[:, :, :-1, :-2]))
        skipcon_3 = torch.cat([encoder_out[2], x_2], 1)
        x_3 = self.imag_act_3(self.imag_bn_3(self.imag_dconv_3(skipcon_3)[:, :, :-1, :-2]))
        skipcon_4 = torch.cat([encoder_out[1], x_3], 1)
        x_4 = self.imag_act_4(self.imag_bn_4(self.imag_dconv_4(skipcon_4)[:, :, :-1, :-2]))
        skipcon_5 = torch.cat([encoder_out[0], x_4], 1)
        x_5 = self.imag_act_5(self.imag_bn_5(self.imag_dconv_5(skipcon_5)[:, :, :-1, :-1]))
        outp = self.inv_flc(x_5)
        return outp

'''
description:注意力机制
return {*}
'''
class AttentionMaskV2(nn.Module):

    def __init__(self, causal):
        super(AttentionMaskV2, self).__init__()
        self.causal = causal

    def lower_triangular_mask(self, shape):
        '''


        Parameters
        ----------
        shape : a tuple of ints

        Returns
        -------
        a square Boolean tensor with the lower triangle being False

        '''
        row_index = torch.cumsum(torch.ones(size=shape), dim=-2)
        col_index = torch.cumsum(torch.ones(size=shape), dim=-1)
        return torch.lt(row_index, col_index)  # lower triangle:True, upper triangle:False

    def merge_masks(self, x, y):

        if x is None: return y
        if y is None: return x
        return torch.logical_and(x, y)

    def forward(self, inp):
        # input (bs, L, ...)
        max_seq_len = inp.shape[1]
        if self.causal == True:
            causal_mask = self.lower_triangular_mask([max_seq_len, max_seq_len])  # (L, l)
            return causal_mask
        else:
            return torch.zeros(size=(max_seq_len, max_seq_len), dtype=torch.float32)

'''
description: 多头注意力
return {*}
'''
class MHAblockV2(nn.Module):

    def __init__(self, d_model, d_ff, n_heads):
        super(MHAblockV2, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

        self.MHA = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads, bias=False)
        self.norm_1 = nn.LayerNorm([self.d_model], eps=1e-6)

        self.fc_1 = nn.Conv1d(self.d_model, self.d_ff, 1)
        self.act = nn.ReLU()
        self.fc_2 = nn.Conv1d(self.d_ff, self.d_model, 1)
        self.norm_2 = nn.LayerNorm([self.d_model], eps=1e-6)

    def forward(self, x, att_mask):
        # x input: (bs, L, d_model)
        x = x.permute(1, 0, 2).contiguous()  # (L, bs, d_model)
        layer_1, _ = self.MHA(x, x, x, attn_mask=att_mask, need_weights=False)  # (L, bs, d_model)
        layer_1 = torch.add(x, layer_1).permute(1, 0, 2).contiguous()  # (L, bs, d_model) ->  (bs, L, d_model)
        layer_1 = self.norm_1(layer_1)  # (bs, L, d_model)

        layer_2 = self.fc_1(layer_1.permute(0, 2, 1).contiguous())  # (bs, d_ff, L)
        layer_2 = self.act(layer_2)  # (bs, d_ff, L)
        layer_2 = self.fc_2(layer_2).permute(0, 2,
                                             1).contiguous()  # (bs, d_ff, L)  -> (bs, d_model, L) -> (bs, L, d_model)
        layer_2 = torch.add(layer_1, layer_2)
        layer_2 = self.norm_2(layer_2)
        return layer_2

'''
description: 位置编码
return {*}
'''
class PositionalEncoding(nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).

    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()

'''
description: 消融实验用DPRNN
return {*}
'''
class DPRNN(nn.Module):
    def __init__(self, numUnits, width, channel, **kwargs):
        super(DPRNN, self).__init__(**kwargs)
        self.numUnits = numUnits

        self.intra_rnn = nn.LSTM(input_size=self.numUnits, hidden_size=self.numUnits // 2, batch_first=True,
                                 bidirectional=True)

        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.InstanceNorm2d(width, eps=1e-8, affine=True)

        self.inter_rnn = nn.LSTM(input_size=self.numUnits, hidden_size=self.numUnits, batch_first=True,
                                 bidirectional=False)

        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)

        self.inter_ln = nn.InstanceNorm2d(channel, eps=1e-8, affine=True)

        self.width = width
        self.channel = channel

    def forward(self, x):
        # x.shape = (Bs, C, T, F)
        self.intra_rnn.flatten_parameters()
        self.inter_rnn.flatten_parameters()
        x = x.permute(0, 2, 3, 1)  # (Bs, T, F, C)
        if not x.is_contiguous():
            x = x.contiguous()
            ## Intra RNN
        intra_LSTM_input = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (Bs*T, F, C)
        intra_LSTM_out = self.intra_rnn(intra_LSTM_input)[0]  # (Bs*T, F, C)
        intra_dense_out = self.intra_fc(intra_LSTM_out)
        intra_ln_input = intra_dense_out.view(x.shape[0], -1, self.width, self.channel)  # (Bs, T, F, C)
        intra_ln_input = intra_ln_input.permute(0, 2, 1, 3)  # (Bs, F, T, C)
        intra_out = self.intra_ln(intra_ln_input)
        intra_out = intra_out.permute(0, 2, 1, 3)  # (Bs, T, F, C)
        intra_out = torch.add(x, intra_out)
        ## Inter RNN
        inter_LSTM_input = intra_out.permute(0, 2, 1, 3)  # (Bs, F, T, C)
        inter_LSTM_input = inter_LSTM_input.contiguous()
        inter_LSTM_input = inter_LSTM_input.view(inter_LSTM_input.shape[0] * inter_LSTM_input.shape[1],
                                                 inter_LSTM_input.shape[2], inter_LSTM_input.shape[3])  # (Bs * F, T, C)
        inter_LSTM_out = self.inter_rnn(inter_LSTM_input)[0]
        inter_dense_out = self.inter_fc(inter_LSTM_out)
        inter_dense_out = inter_dense_out.view(x.shape[0], self.width, -1, self.channel)  # (Bs, F, T, C)
        inter_ln_input = inter_dense_out.permute(0, 3, 2, 1)  # (Bs, C, T, F)
        inter_out = self.inter_ln(inter_ln_input)
        inter_out = inter_out.permute(0, 2, 3, 1)  # (Bs, T, F, C)
        inter_out = torch.add(intra_out, inter_out)
        inter_out = inter_out.permute(0, 3, 1, 2)
        inter_out = inter_out.contiguous()

        return inter_out
'''
description: 第四章模型
return {*}
'''
class DPARN(nn.Module):
    '''
    dual path, intra: MHAnet;  inter: RNN
    '''

    def __init__(self, numUnits, mha_blocks, n_heads, width, channel, device, **kwargs):
        super(DPARN, self).__init__(**kwargs)
        self.numUnits = numUnits
        self.width = width
        self.channel = channel
        self.mha_blocks = mha_blocks
        self.d_model = numUnits
        self.d_ff = 4 * numUnits
        self.n_heads = n_heads
        self.device = device
        self.print = None

        self.pe = PositionalEncoding(input_size=self.d_model)

        self.intra_mha_list = nn.ModuleList(
            [MHAblockV2(self.d_model, self.d_ff, self.n_heads) for _ in range(self.mha_blocks)])

        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.InstanceNorm2d(width, eps=1e-8)

        self.inter_rnn = nn.LSTM(input_size=self.numUnits, hidden_size=self.numUnits, batch_first=True,
                                 bidirectional=False)

        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)

        self.inter_ln = nn.InstanceNorm2d(channel, eps=1e-8)

    def forward(self, x):
        # x.shape = (Bs, C, T, F)
        x = x.permute(0, 2, 3, 1)  # (Bs, T, F, C)
        if not x.is_contiguous():
            x = x.contiguous()
            ## Intra MHA
        intra_MHA_input = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (Bs*T, F, C)
        intra_MHA = intra_MHA_input + self.pe(intra_MHA_input)
        # attention block
        att_mask = AttentionMaskV2(causal=False)(intra_MHA).to(self.device)
        # att_mask = AttentionMaskV2(causal=False)(intra_MHA)
        for att_block in self.intra_mha_list:
            intra_MHA = att_block(intra_MHA, att_mask)  # (bs, L, d_model) = (Bs*T, F, C)

        intra_dense_out = self.intra_fc(intra_MHA)  # (Bs*T, F, C)
        intra_ln_input = intra_dense_out.view(x.shape[0], -1, self.width, self.channel)  # (Bs, T, F, C)
        intra_ln_input = intra_ln_input.permute(0, 2, 1, 3)  # (Bs, F, T, C)
        intra_out = self.intra_ln(intra_ln_input)
        intra_out = intra_out.permute(0, 2, 1, 3)  # (Bs, T, F, C)
        intra_out = torch.add(x, intra_out)
        ## Inter RNN
        inter_LSTM_input = intra_out.permute(0, 2, 1, 3)  # (Bs, F, T, C)
        inter_LSTM_input = inter_LSTM_input.contiguous()
        inter_LSTM_input = inter_LSTM_input.view(inter_LSTM_input.shape[0] * inter_LSTM_input.shape[1],
                                                 inter_LSTM_input.shape[2], inter_LSTM_input.shape[3])  # (Bs * F, T, C)
        inter_LSTM_out = self.inter_rnn(inter_LSTM_input)[0]
        inter_dense_out = self.inter_fc(inter_LSTM_out)
        inter_dense_out = inter_dense_out.view(x.shape[0], self.width, -1, self.channel)  # (Bs, F, T, C)
        inter_ln_input = inter_dense_out.permute(0, 3, 2, 1)  # (Bs, C, T, F)
        inter_out = self.inter_ln(inter_ln_input)
        inter_out = inter_out.permute(0, 2, 3, 1)  # (Bs, T, F, C)
        inter_out = torch.add(intra_out, inter_out)
        inter_out = inter_out.permute(0, 3, 1, 2)  # (Bs, C, T, F)
        inter_out = inter_out.contiguous()

        return inter_out

'''
description: 消融实验 DPRAN
return {*}
'''
class DPRAN(nn.Module):
    '''
    dual path, intra: RNN;  inter: MHA
    '''

    def __init__(self, numUnits, mha_blocks, n_heads, width, channel, device, **kwargs):
        super(DPRAN, self).__init__(**kwargs)
        self.numUnits = numUnits
        self.width = width
        self.channel = channel
        self.mha_blocks = mha_blocks
        self.d_model = numUnits
        self.d_ff = 4 * numUnits
        self.n_heads = n_heads
        self.device = device

        self.intra_rnn = nn.LSTM(input_size=self.numUnits, hidden_size=self.numUnits // 2, batch_first=True,
                                 bidirectional=True)

        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.InstanceNorm2d(width, eps=1e-8)

        self.pe = PositionalEncoding(self.d_model)

        self.inter_mha_list = nn.ModuleList(
            [MHAblockV2(self.d_model, self.d_ff, self.n_heads) for _ in range(self.mha_blocks)])

        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)

        self.inter_ln = nn.InstanceNorm2d(channel, eps=1e-8)

    def forward(self, x):
        # x.shape = (Bs, C, T, F)
        x = x.permute(0, 2, 3, 1)  # (Bs, T, F, C)
        if not x.is_contiguous():
            x = x.contiguous()

            ## Intra RNN
        intra_LSTM_input = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (Bs*T, F, C)
        intra_LSTM_out = self.intra_rnn(intra_LSTM_input)[0]  # (Bs*T, F, C)
        intra_dense_out = self.intra_fc(intra_LSTM_out)
        intra_ln_input = intra_dense_out.view(x.shape[0], -1, self.width, self.channel)  # (Bs, T, F, C)
        intra_ln_input = intra_ln_input.permute(0, 2, 1, 3)  # (Bs, F, T, C)
        intra_out = self.intra_ln(intra_ln_input)
        intra_out = intra_out.permute(0, 2, 1, 3)  # (Bs, T, F, C)
        intra_out = torch.add(x, intra_out)

        ## Inter MHA
        inter_MHA_input = intra_out.permute(0, 2, 1, 3).contiguous()  # (Bs, F, T, C)
        inter_MHA_input = inter_MHA_input.view(inter_MHA_input.shape[0] * inter_MHA_input.shape[1],
                                               inter_MHA_input.shape[2], inter_MHA_input.shape[3])  # (Bs * F, T, C)
        inter_MHA = inter_MHA_input + self.pe(inter_MHA_input)  # (Bs * F, T, C)
        att_mask = AttentionMaskV2(causal=True)(inter_MHA).to(self.device)
        for att_block in self.inter_mha_list:
            inter_MHA = att_block(inter_MHA, att_mask)  # (bs, L, d_model) = (Bs*T, F, C)
        inter_dense_out = self.inter_fc(inter_MHA)
        inter_dense_out = inter_dense_out.view(x.shape[0], self.width, -1, self.channel)  # (Bs, F, T, C)
        inter_ln_input = inter_dense_out.permute(0, 3, 2, 1)  # (Bs, C, T, F)
        inter_out = self.inter_ln(inter_ln_input)
        inter_out = inter_out.permute(0, 2, 3, 1)  # (Bs, T, F, C)
        inter_out = torch.add(intra_out, inter_out)
        inter_out = inter_out.permute(0, 3, 1, 2)
        inter_out = inter_out.contiguous()

        return inter_out

'''
description: 消融实验DPAAN
return {*}
'''
class DPAAN(nn.Module):
    '''
    dual path, intra: MHA;  inter: MHA
    '''

    def __init__(self, numUnits, mha_blocks, n_heads, width, channel, device, **kwargs):
        super(DPAAN, self).__init__(**kwargs)
        self.numUnits = numUnits
        self.width = width
        self.channel = channel
        self.mha_blocks = mha_blocks
        self.d_model = numUnits
        self.d_ff = 4 * numUnits
        self.n_heads = n_heads
        self.device = device

        self.intra_mha_list = nn.ModuleList(
            [MHAblockV2(self.d_model, self.d_ff, self.n_heads) for _ in range(self.mha_blocks[0])])

        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.InstanceNorm2d(width, eps=1e-8)

        self.pe = PositionalEncoding(self.d_model)

        self.inter_mha_list = nn.ModuleList(
            [MHAblockV2(self.d_model, self.d_ff, self.n_heads) for _ in range(self.mha_blocks[1])])

        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)

        self.inter_ln = nn.InstanceNorm2d(channel, eps=1e-8)

    def forward(self, x):
        # x.shape = (Bs, C, T, F)
        x = x.permute(0, 2, 3, 1)  # (Bs, T, F, C)
        if not x.is_contiguous():
            x = x.contiguous()

            ## Intra MHA
        intra_MHA_input = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (Bs*T, F, C)
        intra_MHA = intra_MHA_input + self.pe(intra_MHA_input)  # (Bs * F, T, C)
        # attention block
        att_mask = AttentionMaskV2(causal=False)(intra_MHA).to(self.device)
        for att_block in self.intra_mha_list:
            intra_MHA = att_block(intra_MHA, att_mask)  # (bs, L, d_model) = (Bs*T, F, C)

        intra_dense_out = self.intra_fc(intra_MHA)  # (Bs*T, F, C)
        intra_ln_input = intra_dense_out.view(x.shape[0], -1, self.width, self.channel)  # (Bs, T, F, C)
        intra_ln_input = intra_ln_input.permute(0, 2, 1, 3)  # (Bs, F, T, C)
        intra_out = self.intra_ln(intra_ln_input)
        intra_out = intra_out.permute(0, 2, 1, 3)  # (Bs, T, F, C)
        intra_out = torch.add(x, intra_out)

        ## Inter MHA
        inter_MHA_input = intra_out.permute(0, 2, 1, 3).contiguous()  # (Bs, F, T, C)
        inter_MHA_input = inter_MHA_input.view(inter_MHA_input.shape[0] * inter_MHA_input.shape[1],
                                               inter_MHA_input.shape[2], inter_MHA_input.shape[3])  # (Bs * F, T, C)
        inter_MHA = inter_MHA_input + self.pe(inter_MHA_input)  # (Bs * F, T, C)
        att_mask = AttentionMaskV2(causal=True)(inter_MHA).to(self.device)
        for att_block in self.inter_mha_list:
            inter_MHA = att_block(inter_MHA, att_mask)  # (bs, L, d_model) = (Bs*T, F, C)
        inter_dense_out = self.inter_fc(inter_MHA)
        inter_dense_out = inter_dense_out.view(x.shape[0], self.width, -1, self.channel)  # (Bs, F, T, C)
        inter_ln_input = inter_dense_out.permute(0, 3, 2, 1)  # (Bs, C, T, F)
        inter_out = self.inter_ln(inter_ln_input)
        inter_out = inter_out.permute(0, 2, 3, 1)  # (Bs, T, F, C)
        inter_out = torch.add(intra_out, inter_out)
        inter_out = inter_out.permute(0, 3, 1, 2)
        inter_out = inter_out.contiguous()

        return inter_out

'''
description: 双路径模型 可以选择
return {*}
'''
class DPModel(nn.Module):
    '''
    Dual path model with encoder, decoder, processing block: mhanet or rnn
    '''

    # autoencoder = True
    def __init__(self, model_type, device, num_tscblocks=2):
        super(DPModel, self).__init__()
        self.device = device
        self.encoder = Encoder()
        self.model_type = model_type
        assert model_type in ['DPRAN', 'DPARN', 'DPAAN','DPCRN','DPRANBLOCK1','DPRANBLOCK4','DPRANHEAD1','DPRANHEAD4','DPRANHEAD16'], 'INVALIDE MODEL TYPE.'
        if self.model_type == 'DPARN':
            self.process_model = DPARN(numUnits=80, mha_blocks=2, n_heads=8, width=127, channel=80, device=device)
        if self.model_type == 'DPRAN':
            self.process_model = DPRAN(numUnits=80, mha_blocks=2, n_heads=8, width=127, channel=80, device=device)
        if self.model_type == 'DPRANBLOCK1':
            self.process_model = DPRAN(numUnits=80, mha_blocks=1, n_heads=8, width=127, channel=80, device=device)
        if self.model_type == 'DPRANBLOCK4':
            self.process_model = DPRAN(numUnits=80, mha_blocks=4, n_heads=8, width=127, channel=80, device=device)
        if self.model_type == 'DPRANHEAD1':
            self.process_model = DPRAN(numUnits=80, mha_blocks=2, n_heads=1, width=127, channel=80, device=device)
        if self.model_type == 'DPRANHEAD4':
            self.process_model = DPRAN(numUnits=80, mha_blocks=2, n_heads=4, width=127, channel=80, device=device)
        if self.model_type == 'DPRANHEAD16':
            self.process_model = DPRAN(numUnits=80, mha_blocks=2, n_heads=16, width=127, channel=80, device=device)
        if self.model_type == 'DPAAN':
            self.process_model = DPAAN(numUnits=80, mha_blocks=[2, 2], n_heads=8, width=127, channel=80, device=device)
        if self.model_type == 'DPCRN':
            self.process_model = DPRNN(numUnits=80, width=127, channel=80)
        if self.model_type == 'CONFORMER':
            self.process_model = nn.ModuleList([])
            for i in range(num_tscblocks):
                self.process_model.append(TSConformerBlock())
        self.real_decoder = Real_Decoder()
        self.imag_decoder = Imag_Decoder()

    def init_load(self):
        self.encoder.init_load()

    def forward(self, x):
        # x --> audio batch
        # shape --> [Bs, sequence length]
        encoder_out = self.encoder(x)
        dpath_out = self.process_model(encoder_out[4])
        enh_real = self.real_decoder(dpath_out, encoder_out)
        enh_imag = self.imag_decoder(dpath_out, encoder_out)
        enh_stft = torch.cat([enh_real, enh_imag], 1)  # (Bs, 2, T, F)
        enh_stft = enh_stft.permute(0,3,2,1)

        return enh_stft

    # %%



if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda:0")

    model = DPModel(model_type='DPRAN', device=device)
    # model = DPCONModel()

    model = model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=list(range(2)))
    wave = torch.randn(1,48000)
    x = torch.stft(wave, n_fft=1200, hop_length=600, win_length=1200)  # (Bs, F, T, 2)
    # x = torch.squeeze(wave, dim=1)
    # x = torch.randn(3, 601, 2, 2)
    x = x.to(device)
    flops, params = profile(model, inputs=(x,))
    print(flops)  # 267436032.0   3:73666560.0
    print(params)  # 7823650.0    3:1829010.0
    print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))  # DPARN flops: 3987.65 M, params: 0.84 M   DPRAN  flops: 3855.97 M, params: 0.83 M DPAAN  flops: 4514.34 M, params: 0.89 M DPCRN flops: 3332.57 M, params: 0.77 M
    y = model(x)
    print(y.shape)