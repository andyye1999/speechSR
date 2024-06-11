# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Encodec SEANet-based encoder and decoder implementation."""

import typing as tp

import numpy as np
import torch.nn as nn
import torch
from modules.conv import (
    SConv1d,
    SConvTranspose1d,
)
from modules.lstm import SLSTM
from thop import profile
from local_attention import LocalMHA
from local_attention.transformer import FeedForward, DynamicPositionBias



def exists(val):
    return val is not None

class LocalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        window_size,
        dynamic_pos_bias = False,
        **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.layers = nn.ModuleList([])

        self.pos_bias = None
        if dynamic_pos_bias:
            self.pos_bias = DynamicPositionBias(dim = dim // 2, heads = heads)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LocalMHA(dim = dim, heads = heads, qk_rmsnorm = True, window_size = window_size, use_rotary_pos_emb = not dynamic_pos_bias, use_xpos = True, **kwargs),
                FeedForward(dim = dim)
            ]))

    def forward(self, x):
        w = self.window_size

        attn_bias = self.pos_bias(w, w * 2) if exists(self.pos_bias) else None

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        return x

class ResidualUnit(nn.Module):
    """
    Residual Unit Module
    """
    def __init__(self, channels, nl, dilation, bias=False, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect'):
        super().__init__()

        self.dilated_conv = SConv1d(channels, channels, kernel_size=3, dilation=dilation,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)
        self.pointwise_conv = SConv1d(channels, channels, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)
        self.nl = nl

    def forward(self, x):
        out = x + self.nl(self.pointwise_conv(self.dilated_conv(x)))
        return out


class EncBlock(nn.Module):
    """
    Encoder Block Module
    """
    def __init__(self, out_channels, stride, nl, bias=False, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect'):
        super().__init__()

        self.nl = nl

        self.residuals = nn.Sequential(
            ResidualUnit(channels=out_channels // 2, nl=nl, dilation=1, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
            ResidualUnit(channels=out_channels // 2, nl=nl, dilation=3, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
            ResidualUnit(channels=out_channels // 2, nl=nl, dilation=9, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode))
        self.conv = SConv1d(out_channels // 2, out_channels,
                kernel_size=2 * stride, stride=stride,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

    def forward(self, x):
        out = self.conv(self.residuals(x))
        return out


class DecBlock(nn.Module):
    """
    Decoder Block Module
    """
    def __init__(self, out_channels, stride, nl, bias=False, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect', trim_right_ratio: float = 1.0):
        super().__init__()

        self.nl = nl

        self.residuals = nn.Sequential(
            ResidualUnit(channels=out_channels, nl=nl, dilation=1, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
            ResidualUnit(channels=out_channels, nl=nl, dilation=3, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
            ResidualUnit(channels=out_channels, nl=nl, dilation=9, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode))

        self.conv_trans = SConvTranspose1d(2 * out_channels, out_channels,
                         kernel_size=2 * stride, stride=stride,
                         norm=norm, norm_kwargs=norm_params,
                         causal=causal, trim_right_ratio=trim_right_ratio)

    def forward(self, x, encoder_output):
        x = x + encoder_output
        out = self.residuals(self.nl(self.conv_trans(x)))
        return out


class seanet1(nn.Module):

    def __init__(self, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False, # False
                 pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,stride = [2,2,4,8],first_kernal = 7, latent_kernel = 3, channel = 32):
        """
        Generator of seanet

        """
        super().__init__()


        self.nl = nn.LeakyReLU(negative_slope=0.01)

        self.first_conv = SConv1d(1, channel, kernel_size=first_kernal,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

        self.encoder_blocks = nn.ModuleList(
            [EncBlock(out_channels=channel * 2, stride=stride[0], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 4, stride=stride[1], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 8, stride=stride[2], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 16, stride=stride[3], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode)])

        self.latent_conv = nn.Sequential(self.nl,
                                         SConv1d(channel * 16, channel * 4, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl,
                                         SConv1d(channel * 4, channel * 16, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl)

        self.decoder_blocks = nn.ModuleList(
            [DecBlock(out_channels=channel * 8, stride=stride[3], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 4, stride=stride[2], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 2, stride=stride[1], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel, stride=stride[0], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio)])

        self.last_conv = SConv1d(channel, 1, first_kernal, norm=norm,
                norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)


    def forward(self, cut_audio):
        """
        Forward pass of generator.
        Args:
            cut_audio: in-ear speech signal
        """

        # PQMF analysis, for first band only
        first_band = cut_audio

        # First conv
        x = self.first_conv(first_band)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))
        x4 = self.encoder_blocks[3](self.nl(x3))

        # Latent forward
        x = self.latent_conv(x4)

        # Decoder forward
        x = self.decoder_blocks[0](x, x4)
        x = self.decoder_blocks[1](x, x3)
        x = self.decoder_blocks[2](x, x2)
        x = self.decoder_blocks[3](x, x1)

        # Last conv
        x = self.last_conv(x)

        output = torch.tanh(x + first_band)


        return output

class seanet2(nn.Module):

    def __init__(self, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False, # False
                 pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,stride = [2,4,8],first_kernal = 7, latent_kernel = 3, channel = 32):
        """
        Generator of seanet

        """
        super().__init__()


        self.nl = nn.LeakyReLU(negative_slope=0.01)

        self.first_conv = SConv1d(1, channel, kernel_size=first_kernal,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

        self.encoder_blocks = nn.ModuleList(
            [EncBlock(out_channels=channel * 2, stride=stride[0], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 4, stride=stride[1], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 8, stride=stride[2], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode)])

        self.latent_conv = nn.Sequential(self.nl,
                                         SConv1d(channel * 8, channel * 2, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl,
                                         SConv1d(channel * 2, channel * 8, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl)

        self.decoder_blocks = nn.ModuleList(
            [
             DecBlock(out_channels=channel * 4, stride=stride[2], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 2, stride=stride[1], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel, stride=stride[0], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio)])

        self.last_conv = SConv1d(channel, 1, first_kernal, norm=norm,
                norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)


    def forward(self, cut_audio):
        """
        Forward pass of generator.
        Args:
            cut_audio: in-ear speech signal
        """

        # PQMF analysis, for first band only
        first_band = cut_audio

        # First conv
        x = self.first_conv(first_band)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))
        # x4 = self.encoder_blocks[3](self.nl(x3))

        # Latent forward
        x = self.latent_conv(x3)

        # Decoder forward
        # x = self.decoder_blocks[0](x, x4)
        x = self.decoder_blocks[0](x, x3)
        x = self.decoder_blocks[1](x, x2)
        x = self.decoder_blocks[2](x, x1)

        # Last conv
        x = self.last_conv(x)

        output = torch.tanh(x + first_band)


        return output


class seanet3(nn.Module):

    def __init__(self, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False, # False
                 pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,stride = [2,2,4,8],first_kernal = 7, latent_kernel = 3, channel = 16):
        """
        Generator of seanet

        """
        super().__init__()


        self.nl = nn.LeakyReLU(negative_slope=0.01)

        self.first_conv = SConv1d(1, channel, kernel_size=first_kernal,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

        self.encoder_blocks = nn.ModuleList(
            [EncBlock(out_channels=channel * 2, stride=stride[0], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 4, stride=stride[1], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 8, stride=stride[2], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 16, stride=stride[3], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode)])

        self.latent_conv = nn.Sequential(self.nl,
                                         SConv1d(channel * 16, channel * 4, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl,
                                         SConv1d(channel * 4, channel * 16, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl)

        self.decoder_blocks = nn.ModuleList(
            [DecBlock(out_channels=channel * 8, stride=stride[3], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 4, stride=stride[2], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 2, stride=stride[1], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel, stride=stride[0], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio)])

        self.last_conv = SConv1d(channel, 1, first_kernal, norm=norm,
                norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)


    def forward(self, cut_audio):
        """
        Forward pass of generator.
        Args:
            cut_audio: in-ear speech signal
        """

        # PQMF analysis, for first band only
        first_band = cut_audio

        # First conv
        x = self.first_conv(first_band)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))
        x4 = self.encoder_blocks[3](self.nl(x3))

        # Latent forward
        x = self.latent_conv(x4)

        # Decoder forward
        x = self.decoder_blocks[0](x, x4)
        x = self.decoder_blocks[1](x, x3)
        x = self.decoder_blocks[2](x, x2)
        x = self.decoder_blocks[3](x, x1)

        # Last conv
        x = self.last_conv(x)

        output = torch.tanh(x + first_band)


        return output

class seanet4(nn.Module):

    def __init__(self, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False, # False
                 pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,stride = [2,2,4,8],first_kernal = 7, latent_kernel = 3, channel = 24):
        """
        Generator of seanet

        """
        super().__init__()


        self.nl = nn.LeakyReLU(negative_slope=0.01)

        self.first_conv = SConv1d(1, channel, kernel_size=first_kernal,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

        self.encoder_blocks = nn.ModuleList(
            [EncBlock(out_channels=channel * 2, stride=stride[0], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 4, stride=stride[1], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 8, stride=stride[2], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 16, stride=stride[3], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode)])

        self.latent_conv = nn.Sequential(self.nl,
                                         SConv1d(channel * 16, channel * 4, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl,
                                         SConv1d(channel * 4, channel * 16, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl)

        self.decoder_blocks = nn.ModuleList(
            [DecBlock(out_channels=channel * 8, stride=stride[3], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 4, stride=stride[2], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 2, stride=stride[1], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel, stride=stride[0], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio)])

        self.last_conv = SConv1d(channel, 1, first_kernal, norm=norm,
                norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)


    def forward(self, cut_audio):
        """
        Forward pass of generator.
        Args:
            cut_audio: in-ear speech signal
        """

        # PQMF analysis, for first band only
        first_band = cut_audio

        # First conv
        x = self.first_conv(first_band)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))
        x4 = self.encoder_blocks[3](self.nl(x3))

        # Latent forward
        x = self.latent_conv(x4)

        # Decoder forward
        x = self.decoder_blocks[0](x, x4)
        x = self.decoder_blocks[1](x, x3)
        x = self.decoder_blocks[2](x, x2)
        x = self.decoder_blocks[3](x, x1)

        # Last conv
        x = self.last_conv(x)

        output = torch.tanh(x + first_band)


        return output

class seanet5(nn.Module):

    def __init__(self, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False, # False
                 pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,stride = [2,2,4,5],first_kernal = 7, latent_kernel = 3, channel = 16):
        """
        Generator of seanet

        """
        super().__init__()


        self.nl = nn.LeakyReLU(negative_slope=0.01)

        self.first_conv = SConv1d(1, channel, kernel_size=first_kernal,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

        self.encoder_blocks = nn.ModuleList(
            [EncBlock(out_channels=channel * 2, stride=stride[0], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 4, stride=stride[1], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 8, stride=stride[2], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 16, stride=stride[3], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode)])

        self.latent_conv = nn.Sequential(self.nl,
                                         SConv1d(channel * 16, channel * 4, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl,
                                         SConv1d(channel * 4, channel * 16, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl)

        self.decoder_blocks = nn.ModuleList(
            [DecBlock(out_channels=channel * 8, stride=stride[3], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 4, stride=stride[2], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 2, stride=stride[1], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel, stride=stride[0], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio)])

        self.last_conv = SConv1d(channel, 1, first_kernal, norm=norm,
                norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)


    def forward(self, cut_audio):
        """
        Forward pass of generator.
        Args:
            cut_audio: in-ear speech signal
        """

        # PQMF analysis, for first band only
        first_band = cut_audio

        # First conv
        x = self.first_conv(first_band)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))
        x4 = self.encoder_blocks[3](self.nl(x3))

        # Latent forward
        x = self.latent_conv(x4)

        # Decoder forward
        x = self.decoder_blocks[0](x, x4)
        x = self.decoder_blocks[1](x, x3)
        x = self.decoder_blocks[2](x, x2)
        x = self.decoder_blocks[3](x, x1)

        # Last conv
        x = self.last_conv(x)

        output = torch.tanh(x + first_band)


        return output

class seanetbwe(nn.Module):

    def __init__(self, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False, # False
                 pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,stride = [2,4,8],first_kernal = 7, latent_kernel = 7, channel = 16):
        """
        Generator of seanet

        """
        super().__init__()


        self.nl = nn.LeakyReLU(negative_slope=0.01)

        self.first_conv = SConv1d(1, channel, kernel_size=first_kernal,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

        self.encoder_blocks = nn.ModuleList(
            [EncBlock(out_channels=channel * 2, stride=stride[0], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 4, stride=stride[1], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 8, stride=stride[2], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode)])

        self.latent_conv = nn.Sequential(self.nl,
                                         SConv1d(channel * 8, channel * 2, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl,
                                         SConv1d(channel * 2, channel * 8, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl)

        self.decoder_blocks = nn.ModuleList(
            [
             DecBlock(out_channels=channel * 4, stride=stride[2], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 2, stride=stride[1], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel, stride=stride[0], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio)])

        self.last_conv = SConv1d(channel, 1, first_kernal, norm=norm,
                norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)


    def forward(self, cut_audio):
        """
        Forward pass of generator.
        Args:
            cut_audio: in-ear speech signal
        """

        # PQMF analysis, for first band only
        first_band = cut_audio

        # First conv
        x = self.first_conv(first_band)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))
        # x4 = self.encoder_blocks[3](self.nl(x3))

        # Latent forward
        x = self.latent_conv(x3)

        # Decoder forward
        # x = self.decoder_blocks[0](x, x4)
        x = self.decoder_blocks[0](x, x3)
        x = self.decoder_blocks[1](x, x2)
        x = self.decoder_blocks[2](x, x1)

        # Last conv
        x = self.last_conv(x)

        output = torch.tanh(x + first_band)


        return output

class seanetbwe2(nn.Module):

    def __init__(self, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False, # False
                 pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,stride = [2,4,8],first_kernal = 7, latent_kernel = 7, channel = 32):
        """
        Generator of seanet

        """
        super().__init__()


        self.nl = nn.LeakyReLU(negative_slope=0.01)

        self.first_conv = SConv1d(1, channel, kernel_size=first_kernal,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

        self.encoder_blocks = nn.ModuleList(
            [EncBlock(out_channels=channel * 2, stride=stride[0], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 4, stride=stride[1], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 8, stride=stride[2], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode)])

        self.latent_conv = nn.Sequential(self.nl,
                                         SConv1d(channel * 8, channel * 2, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl,
                                         SConv1d(channel * 2, channel * 8, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl)

        self.decoder_blocks = nn.ModuleList(
            [
             DecBlock(out_channels=channel * 4, stride=stride[2], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 2, stride=stride[1], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel, stride=stride[0], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio)])

        self.last_conv = SConv1d(channel, 1, first_kernal, norm=norm,
                norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)


    def forward(self, cut_audio):
        """
        Forward pass of generator.
        Args:
            cut_audio: in-ear speech signal
        """

        # PQMF analysis, for first band only
        first_band = cut_audio

        # First conv
        x = self.first_conv(first_band)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))
        # x4 = self.encoder_blocks[3](self.nl(x3))

        # Latent forward
        x = self.latent_conv(x3)

        # Decoder forward
        # x = self.decoder_blocks[0](x, x4)
        x = self.decoder_blocks[0](x, x3)
        x = self.decoder_blocks[1](x, x2)
        x = self.decoder_blocks[2](x, x1)

        # Last conv
        x = self.last_conv(x)

        output = torch.tanh(x + first_band)


        return output

class seanetbweattention(nn.Module):

    def __init__(self, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False, # False
                 pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,stride = [2,4,8],first_kernal = 3, latent_kernel = 3, channel = 32):
        """
        Generator of seanet

        """
        super().__init__()


        self.nl = nn.LeakyReLU(negative_slope=0.01)

        attn_kwargs = dict(
            dim=channel*8,
            dim_head=64,
            heads=8,
            depth=1,
            window_size=128,
            xpos_scale_base=None,
            dynamic_pos_bias=False,
            prenorm=True,
            causal=True
        )
        self.encoder_attn = LocalTransformer(**attn_kwargs)
        self.decoder_attn = LocalTransformer(**attn_kwargs)

        self.first_conv = SConv1d(1, channel, kernel_size=first_kernal,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

        self.encoder_blocks = nn.ModuleList(
            [EncBlock(out_channels=channel * 2, stride=stride[0], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 4, stride=stride[1], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 8, stride=stride[2], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode)])

        self.latent_conv = nn.Sequential(self.nl,
                                         SConv1d(channel * 8, channel * 2, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl,
                                         SConv1d(channel * 2, channel * 8, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl)

        self.decoder_blocks = nn.ModuleList(
            [
             DecBlock(out_channels=channel * 4, stride=stride[2], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 2, stride=stride[1], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel, stride=stride[0], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio)])

        self.last_conv = SConv1d(channel, 1, first_kernal, norm=norm,
                norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)


    def forward(self, cut_audio):
        """
        Forward pass of generator.
        Args:
            cut_audio: in-ear speech signal
        """

        # PQMF analysis, for first band only
        first_band = cut_audio

        # First conv
        x = self.first_conv(first_band)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))
        # x4 = self.encoder_blocks[3](self.nl(x3))

        # Latent forward
        x = self.latent_conv(x3)

        x = x.permute(0, 2, 1).contiguous()  # b,c,n -> b ,n,c
        x = self.encoder_attn(x)
        x = x.permute(0, 2, 1).contiguous()

        # Decoder forward
        # x = self.decoder_blocks[0](x, x4)
        x = self.decoder_blocks[0](x, x3)
        x = self.decoder_blocks[1](x, x2)
        x = self.decoder_blocks[2](x, x1)

        # Last conv
        x = self.last_conv(x)

        output = torch.tanh(x + first_band)


        return output


class seanetbweattention1(nn.Module):

    def __init__(self, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False, # False
                 pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,stride = [2,4,8],first_kernal = 7, latent_kernel = 7, channel = 16):
        """
        Generator of seanet

        """
        super().__init__()


        self.nl = nn.LeakyReLU(negative_slope=0.01)

        attn_kwargs = dict(
            dim=channel*8,
            dim_head=64,
            heads=8,
            depth=1,
            window_size=128,
            xpos_scale_base=None,
            dynamic_pos_bias=False,
            prenorm=True,
            causal=True
        )
        self.encoder_attn = LocalTransformer(**attn_kwargs)
        self.decoder_attn = LocalTransformer(**attn_kwargs)

        self.first_conv = SConv1d(1, channel, kernel_size=first_kernal,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

        self.encoder_blocks = nn.ModuleList(
            [EncBlock(out_channels=channel * 2, stride=stride[0], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 4, stride=stride[1], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 8, stride=stride[2], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode)])

        self.latent_conv = nn.Sequential(self.nl,
                                         SConv1d(channel * 8, channel * 2, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl,
                                         SConv1d(channel * 2, channel * 8, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl)

        self.decoder_blocks = nn.ModuleList(
            [
             DecBlock(out_channels=channel * 4, stride=stride[2], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 2, stride=stride[1], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel, stride=stride[0], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio)])

        self.last_conv = SConv1d(channel, 1, first_kernal, norm=norm,
                norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)


    def forward(self, cut_audio):
        """
        Forward pass of generator.
        Args:
            cut_audio: in-ear speech signal
        """

        # PQMF analysis, for first band only
        first_band = cut_audio

        # First conv
        x = self.first_conv(first_band)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))
        # x4 = self.encoder_blocks[3](self.nl(x3))

        # Latent forward
        x = self.latent_conv(x3)

        x = x.permute(0, 2, 1).contiguous()  # b,c,n -> b ,n,c
        x = self.encoder_attn(x)
        x = x.permute(0, 2, 1).contiguous()

        # Decoder forward
        # x = self.decoder_blocks[0](x, x4)
        x = self.decoder_blocks[0](x, x3)
        x = self.decoder_blocks[1](x, x2)
        x = self.decoder_blocks[2](x, x1)

        # Last conv
        x = self.last_conv(x)

        output = torch.tanh(x + first_band)


        return output

class seanetbweattention4(nn.Module):

    def __init__(self, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False, # False
                 pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,stride = [2,4,8],first_kernal = 7, latent_kernel = 7, channel = 32):
        """
        Generator of seanet

        """
        super().__init__()


        self.nl = nn.LeakyReLU(negative_slope=0.01)

        attn_kwargs = dict(
            dim=channel*8,
            dim_head=64,
            heads=8,
            depth=1,
            window_size=128,
            xpos_scale_base=None,
            dynamic_pos_bias=False,
            prenorm=True,
            causal=True
        )
        self.encoder_attn = LocalTransformer(**attn_kwargs)
        self.decoder_attn = LocalTransformer(**attn_kwargs)

        self.first_conv = SConv1d(1, channel, kernel_size=first_kernal,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

        self.encoder_blocks = nn.ModuleList(
            [EncBlock(out_channels=channel * 2, stride=stride[0], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 4, stride=stride[1], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 8, stride=stride[2], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode)])

        self.latent_conv = nn.Sequential(self.nl,
                                         SConv1d(channel * 8, channel * 2, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl,
                                         SConv1d(channel * 2, channel * 8, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl)

        self.decoder_blocks = nn.ModuleList(
            [
             DecBlock(out_channels=channel * 4, stride=stride[2], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 2, stride=stride[1], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel, stride=stride[0], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio)])

        self.last_conv = SConv1d(channel, 1, first_kernal, norm=norm,
                norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)


    def forward(self, cut_audio):
        """
        Forward pass of generator.
        Args:
            cut_audio: in-ear speech signal
        """

        # PQMF analysis, for first band only
        first_band = cut_audio

        # First conv
        x = self.first_conv(first_band)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))
        # x4 = self.encoder_blocks[3](self.nl(x3))

        # Latent forward
        x = self.latent_conv(x3)

        x = x.permute(0, 2, 1).contiguous()  # b,c,n -> b ,n,c
        x = self.encoder_attn(x)
        x = x.permute(0, 2, 1).contiguous()

        # Decoder forward
        # x = self.decoder_blocks[0](x, x4)
        x = self.decoder_blocks[0](x, x3)
        x = self.decoder_blocks[1](x, x2)
        x = self.decoder_blocks[2](x, x1)

        # Last conv
        x = self.last_conv(x)

        output = torch.tanh(x + first_band)


        return output

class seanetbweattention2(nn.Module):

    def __init__(self, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False, # False
                 pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,stride = [2,4,5,8],first_kernal = 7, latent_kernel = 7, channel = 32):
        """
        Generator of seanet

        """
        super().__init__()


        self.nl = nn.LeakyReLU(negative_slope=0.01)

        self.first_conv = SConv1d(1, channel, kernel_size=first_kernal,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

        self.encoder_blocks = nn.ModuleList(
            [EncBlock(out_channels=channel * 2, stride=stride[0], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 4, stride=stride[1], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 8, stride=stride[2], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 16, stride=stride[3], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode)])

        attn_kwargs = dict(
            dim=channel * 16,
            dim_head=64,
            heads=8,
            depth=1,
            window_size=128,
            xpos_scale_base=None,
            dynamic_pos_bias=False,
            prenorm=True,
            causal=True
        )
        self.encoder_attn = LocalTransformer(**attn_kwargs)
        self.decoder_attn = LocalTransformer(**attn_kwargs)


        self.latent_conv = nn.Sequential(self.nl,
                                         SConv1d(channel * 16, channel * 4, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl,
                                         SConv1d(channel * 4, channel * 16, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl)

        self.decoder_blocks = nn.ModuleList(
            [DecBlock(out_channels=channel * 8, stride=stride[3], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 4, stride=stride[2], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 2, stride=stride[1], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel, stride=stride[0], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio)])

        self.last_conv = SConv1d(channel, 1, first_kernal, norm=norm,
                norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)


    def forward(self, cut_audio):
        """
        Forward pass of generator.
        Args:
            cut_audio: in-ear speech signal
        """

        # PQMF analysis, for first band only
        first_band = cut_audio

        # First conv
        x = self.first_conv(first_band)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))
        x4 = self.encoder_blocks[3](self.nl(x3))

        # Latent forward
        x = self.latent_conv(x4)

        x = x.permute(0, 2, 1).contiguous()  # b,c,n -> b ,n,c
        x = self.encoder_attn(x)
        x = x.permute(0, 2, 1).contiguous()

        # Decoder forward
        x = self.decoder_blocks[0](x, x4)
        x = self.decoder_blocks[1](x, x3)
        x = self.decoder_blocks[2](x, x2)
        x = self.decoder_blocks[3](x, x1)

        # Last conv
        x = self.last_conv(x)

        output = torch.tanh(x + first_band)


        return output

class seanetbweattention3(nn.Module):

    def __init__(self, norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False, # False
                 pad_mode: str = 'reflect', trim_right_ratio: float = 1.0,stride = [2,2,8,8],first_kernal = 7, latent_kernel = 7, channel = 32):
        """
        Generator of seanet

        """
        super().__init__()


        self.nl = nn.LeakyReLU(negative_slope=0.01)

        self.first_conv = SConv1d(1, channel, kernel_size=first_kernal,
                norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)

        self.encoder_blocks = nn.ModuleList(
            [EncBlock(out_channels=channel * 2, stride=stride[0], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 4, stride=stride[1], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 8, stride=stride[2], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode),
             EncBlock(out_channels=channel * 16, stride=stride[3], nl=self.nl, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode)])

        attn_kwargs = dict(
            dim=channel * 16,
            dim_head=64,
            heads=8,
            depth=1,
            window_size=128,
            xpos_scale_base=None,
            dynamic_pos_bias=False,
            prenorm=True,
            causal=True
        )
        self.encoder_attn = LocalTransformer(**attn_kwargs)
        self.decoder_attn = LocalTransformer(**attn_kwargs)


        self.latent_conv = nn.Sequential(self.nl,
                                         SConv1d(channel * 16, channel * 4, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl,
                                         SConv1d(channel * 4, channel * 16, latent_kernel, norm=norm,
                                                 norm_kwargs=norm_params,
                                                 causal=causal, pad_mode=pad_mode),
                                         self.nl)

        self.decoder_blocks = nn.ModuleList(
            [DecBlock(out_channels=channel * 8, stride=stride[3], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 4, stride=stride[2], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel * 2, stride=stride[1], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
             DecBlock(out_channels=channel, stride=stride[0], nl=self.nl,
                                 norm=norm, norm_params=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio)])

        self.last_conv = SConv1d(channel, 1, first_kernal, norm=norm,
                norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode)


    def forward(self, cut_audio):
        """
        Forward pass of generator.
        Args:
            cut_audio: in-ear speech signal
        """

        # PQMF analysis, for first band only
        first_band = cut_audio

        # First conv
        x = self.first_conv(first_band)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))
        x4 = self.encoder_blocks[3](self.nl(x3))

        # Latent forward
        x = self.latent_conv(x4)

        x = x.permute(0, 2, 1).contiguous()  # b,c,n -> b ,n,c
        x = self.encoder_attn(x)
        x = x.permute(0, 2, 1).contiguous()

        # Decoder forward
        x = self.decoder_blocks[0](x, x4)
        x = self.decoder_blocks[1](x, x3)
        x = self.decoder_blocks[2](x, x2)
        x = self.decoder_blocks[3](x, x1)

        # Last conv
        x = self.last_conv(x)

        output = torch.tanh(x + first_band)


        return output

# def test():
#     import torch
#     encoder = SEANetEncoder()
#     print(encoder)
#     decoder = SEANetDecoder()
#     x = torch.randn(1, 1, 24000)
#     z = encoder(x)
#     print('z ', z.shape)
#     # assert 1==2
#     assert list(z.shape) == [1, 128, 75], z.shape
#     y = decoder(z)
#     assert y.shape == x.shape, (x.shape, y.shape)


if __name__ == '__main__':
    x = torch.randn(1,1,48000)
    model = seanetbwe()
    y = model(x)
    print(y.shape)
    # Number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"pytorch_total_params: {pytorch_total_params * 1e-6:.2f} Millions")

    flops, params = profile(model, inputs=(x,))
    print(flops) # 267436032.0   3:73666560.0
    print(params) # 7823650.0    3:1829010.0
    print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))  # seanetbwe flops: 3053.15 M, params: 0.49 M  seanetbweattention1  flops: 3348.73 M, params: 0.88 M