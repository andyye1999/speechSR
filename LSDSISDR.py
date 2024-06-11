'''
Author: yehongcen && as_yhc@163.com
Date: 2024-04-23 15:36:38
LastEditors: redust as_yhc@163.com
LastEditTime: 2024-05-23 16:08:27
FilePath: \speechSR\LSDSISDR.py
Description: 测试数据的性能指标SNR和SI-SNR PESQ
Copyright (c) 2024 by yehongcen, All Rights Reserved. 
'''
import numpy as np
import librosa
import os
import torch
import pesq
from pystoi import stoi
from torchlibrosa.stft import STFT
'''
description: L2范数
param {*} s1
param {*} s2
return {*}
'''
def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm

'''
description: 计算SI-SNR
param {*} s1
param {*} s2
param {*} eps
return {*}
'''
def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2) # 计算两个语音之间的L2范数
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2 # 计算目标语音的SI-SNR
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10(target_norm / (noise_norm + eps) + eps) # 计算SI-SNR
    return torch.mean(snr)


'''
description: 计算SI-SNR np版本
param {*} target
param {*} preds
return {*}
'''
def SI_SDR(target, preds):
    EPS = 1e-8
    alpha = (np.sum(preds * target, axis=-1, keepdims=True) + EPS) / (np.sum(target ** 2, axis=-1, keepdims=True) + EPS)
    target_scaled = alpha * target
    noise = target_scaled - preds
    si_sdr_value = (np.sum(target_scaled ** 2, axis=-1) + EPS) / (np.sum(noise ** 2, axis=-1) + EPS)
    si_sdr_value = 10 * np.log10(si_sdr_value)
    return si_sdr_value

'''
description: 计算能量
param {*} x
param {*} nfft
return {*}
'''
def get_power(x, nfft):
    S = librosa.stft(x, n_fft=nfft)
    S = np.log(np.abs(S) ** 2 + 1e-8)
    return S

'''
description: 计算LSD
param {*} x_hr
param {*} x_pr
return {*}
'''
def LSD(x_hr, x_pr):
    S1 = get_power(x_hr, nfft=2048) # 2048
    S2 = get_power(x_pr, nfft=2048) 
    lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    S1 = S1[-(len(S1) - 1) // 2:, :]
    S2 = S2[-(len(S2) - 1) // 2:, :]
    lsd_high = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    return lsd, lsd_high

'''
description: 计算LSD
param {*} y_pred
param {*} y_true
return {*}
'''
def LSD_NEW(y_pred,y_true):
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true) # 转pytorch
    y_pred = torch.squeeze(y_pred)
    y_true = torch.squeeze(y_true)
    y_pred = torch.unsqueeze(y_pred,dim=0)
    y_true = torch.unsqueeze(y_true,dim=0)

    y_pred = y_pred.to(torch.device("cpu"))  # [1, 1, T]
    y_true = y_true.to(torch.device("cpu"))
    stft = STFT(n_fft=2048, hop_length=512,
                win_length=2048, window='hann', center=True,
                pad_mode='reflect') # 计算STFT
    pred_stft_real, pred_stft_imag  = stft(y_pred.float()) # 计算实部和虚部
    true_stft_real, true_stft_imag  = stft(y_true.float()) 
    # pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
    # true_stft_real, true_stft_imag = true_stft[:, :, :, 0], true_stft[:, :, :, 1]
    pred_mag = torch.clamp(pred_stft_real ** 2 + pred_stft_imag ** 2, 1e-8, np.inf) ** 0.5 # 幅度
    true_mag = torch.clamp(true_stft_real ** 2 + true_stft_imag ** 2, 1e-8, np.inf) ** 0.5 
    lsd = torch.log10((true_mag ** 2 / ((pred_mag + 1e-8) ** 2)) + 1e-8) ** 2 
    lsd = torch.mean(torch.mean(lsd, dim=3) ** 0.5, dim=2)
    lsd = lsd.detach().cpu().numpy()
    return lsd[..., None, None]

'''
description: 计算SNR
param {*} preds
param {*} target
return {*}
'''
def SNR(preds, target):
    EPS = 1e-8
    # alpha = (np.sum(preds * target, axis=-1, keepdims=True) + EPS) / (np.sum(target ** 2, axis=-1, keepdims=True) + EPS)
    # target_scaled = alpha * target
    noise = target - preds
    si_sdr_value = (np.sum(target ** 2, axis=-1) + EPS) / (np.sum(noise ** 2, axis=-1) + EPS)
    si_sdr_value = 10 * np.log10(si_sdr_value)
    return si_sdr_value

# 读取文件夹1中的音频文件
folder1 = 'D:\\yhc\\BWE\\enhanced\\show\\original'
audio_files1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.wav')]

# 读取文件夹2中的音频文件
folder2 = 'D:\\yhc\\BWE\\enhanced\\show\\dpcra'
audio_files2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith('.wav')]

# 计算SI_SDR和LSD
si_sdr_list = []
lsd_list = []
lsd_high_list = []
pesq_list = []
stoi_list = []
snr_list=[]
for audio_file1, audio_file2 in zip(audio_files1, audio_files2): # 计算每个音频的指标
    x_hr, sr = librosa.load(audio_file1, sr=None)
    x_pr, sr = librosa.load(audio_file2, sr=None)
    if len(x_hr) > len(x_pr):
        x_hr = x_hr[:len(x_pr)]
    else:
        x_pr = x_pr[:len(x_hr)]
    x_hr1 = torch.from_numpy(x_hr) # 转pytorch
    x_pr1 = torch.from_numpy(x_pr)
    x_hr1 = torch.unsqueeze(x_hr1, 0)
    x_pr1 = torch.unsqueeze(x_pr1, 0)
    torch_si_snr = si_snr(x_pr1,x_hr1) # 计算torch的SI-SNR
    si_sdr_value = SI_SDR(x_hr, x_pr) # 计算np的SI-SNR
    snr_value = SNR(x_pr, x_hr) # 计算np的SNR
    # lsd, lsd_high = LSD(x_hr, x_pr)
    lsdnew = LSD_NEW(x_hr, x_pr) # 计算LSD
    # pesq_value = pesq.pesq(sr, x_hr, x_pr, 'wb')
    # stoi_value = stoi(x_hr, x_pr, sr, extended=False)
    si_sdr_list.append(si_sdr_value)
    lsd_list.append(lsdnew)
    snr_list.append(snr_value)
    # lsd_high_list.append(lsd_high)
    # pesq_list.append(pesq_value)
    # stoi_list.append(stoi_value)
    print(f"audio_file1: { audio_file1 }, SI_SDR: {si_sdr_value}, LSD: {lsdnew}, torch_si_snr: {torch_si_snr}, SNR: {snr_value}")
    # print(f"PESQ: {pesq_value}, STOI: {stoi_value}")

# 计算SI_SDR和LSD的平均值
si_sdr_mean = np.mean(si_sdr_list)
lsd_mean = np.mean(lsd_list)
lsd_high_mean = np.mean(lsd_high_list)
pesq_mean = np.mean(pesq_list)
stoi_mean = np.mean(stoi_list)
snr_mean = np.mean(snr_list)

print(f"SI_SDR平均值: {si_sdr_mean}, SNR平均值: {snr_mean}, LSD平均值: {lsd_mean}, LSD_high平均值: {lsd_high_mean}, PESQ平均值: {pesq_mean}, STOI平均值: {stoi_mean}")
# 12-48
# aero SI_SDR平均值: 27.406809980188513, SNR平均值: 27.37594842467282, LSD平均值: 0.8654531240463257
# seanet SI_SDR平均值: 26.14011780542784, SNR平均值: 26.13553496951979, LSD平均值: 0.850429356098175
# cibrn SI_SDR平均值: 26.762419954900146, SNR平均值: 26.76959830265794, LSD平均值: 0.8459914326667786
# nuwave  SI_SDR平均值: 28.57009358287136, SNR平均值: 3.883158610516144, LSD平均值: 1.022631287574768
# 16-48
# dpcra SI_SDR平均值: 26.87475456419934, SNR平均值: 26.881907241439475, LSD平均值: 0.839799702167511