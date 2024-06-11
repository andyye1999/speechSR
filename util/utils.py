'''
Author: yehongcen && as_yhc@163.com
Date: 2024-04-23 15:36:51
LastEditors: redust as_yhc@163.com
LastEditTime: 2024-05-23 16:11:53
FilePath: \speechSR\util\utils.py
Description: 一些补充函数
Copyright (c) 2024 by yehongcen, All Rights Reserved. 
'''
import importlib
import time
import os
import random
import torch
from pesq import pesq
import numpy as np
from pystoi.stoi import stoi
import librosa
from torchlibrosa.stft import STFT
'''
description: 加载checkpoint
param {*} checkpoint_path
param {*} device
return {*}
'''
def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(checkpoint_path, map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]


'''
description: 生成空文件夹
param {*} dirs
param {*} resume
return {*}
'''
def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def initialize_config(module_cfg, pass_args=True):
    """According to config items, load specific module dynamically with params.
    e.g., Config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])



def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    return pesq(sr, clean_signal, noisy_signal, "wb")

def compute_PESQ8k(clean_signal, noisy_signal, sr=8000):
    return pesq(sr, clean_signal, noisy_signal, "nb")

def z_score(m):
    mean = np.mean(m)
    std_var = np.std(m)
    return (m - mean) / std_var, mean, std_var


def reverse_z_score(m, mean, std_var):
    return m * std_var + mean


def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min


def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min


'''
description: 将两个音频文件对齐
param {*} data_a
param {*} data_b
param {*} sample_length
return {*}
'''
def sample_fixed_length_data_aligned(data_a, data_b, sample_length):
    """sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    frames_total = len(data_a)

    start = np.random.randint(frames_total - sample_length + 1)
    # print(f"Random crop from: {start}")
    end = start + sample_length

    return data_a[start:end], data_b[start:end]

def sample_fixed_length_data_alignedss(data_a, sample_length):
    """sample with fixed length from two dataset
    """
    # assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    frames_total = len(data_a)

    start = np.random.randint(frames_total - sample_length + 1)
    # print(f"Random crop from: {start}")
    end = start + sample_length

    return data_a[start:end]

'''
description: STOI
param {*} clean_signal
param {*} noisy_signal
param {*} sr
return {*}
'''
def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)

'''
description: 能量
param {*} x
param {*} nfft
return {*}
'''
def get_power(x, nfft):
    S = librosa.stft(x, n_fft=nfft)
    S = np.log(np.abs(S) ** 2 + 1e-8)
    return S




hf = int(1025 * (48000 / 16000))

'''
description: LSD
param {*} y_pred
param {*} y_true
return {*}
'''
def LSD_NEW(y_pred,y_true):
    '''
    description: 计算LSD
    return {*}
    '''
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    y_pred = torch.squeeze(y_pred)
    y_true = torch.squeeze(y_true)
    y_pred = torch.unsqueeze(y_pred,dim=0)
    y_true = torch.unsqueeze(y_true,dim=0)

    y_pred = y_pred.to(torch.device("cpu"))  # [1, 1, T]
    y_true = y_true.to(torch.device("cpu"))
    stft = STFT(n_fft=2048, hop_length=512,
                win_length=2048, window='hann', center=True,
                pad_mode='reflect') # STFT
    pred_stft_real, pred_stft_imag  = stft(y_pred.float())
    true_stft_real, true_stft_imag  = stft(y_true.float())
    # pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
    # true_stft_real, true_stft_imag = true_stft[:, :, :, 0], true_stft[:, :, :, 1]
    pred_mag = torch.clamp(pred_stft_real ** 2 + pred_stft_imag ** 2, 1e-8, np.inf) ** 0.5 # 幅度
    true_mag = torch.clamp(true_stft_real ** 2 + true_stft_imag ** 2, 1e-8, np.inf) ** 0.5
    lsd = torch.log10((true_mag ** 2 / ((pred_mag + 1e-8) ** 2)) + 1e-8) ** 2 # log
    lsd = torch.mean(torch.mean(lsd, dim=3) ** 0.5, dim=2)
    lsd = lsd.detach().cpu().numpy()
    return lsd[..., None, None]


def print_tensor_info(tensor, flag="Tensor"):
    floor_tensor = lambda float_tensor: int(float(float_tensor) * 1000) / 1000
    print(flag)
    print(
        f"\tmax: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}")

def sample_pad(data_a, data_b, sample_length):
    """sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    # data_a, data_b = normalize_data(data_a,data_b)
    frames_total = len(data_a)

    # 计算需要填充的帧数
    remainder = frames_total % sample_length
    if remainder != 0:
        num_frames_to_pad = sample_length - remainder
        # 使用零填充数据
        data_a = np.pad(data_a, (0, num_frames_to_pad))
        data_b = np.pad(data_b, (0, num_frames_to_pad))
        frames_total += num_frames_to_pad

    # start = np.random.randint(frames_total - sample_length + 1)
    # # print(f"Random crop from: {start}")
    # end = start + sample_length

    return data_a, data_b


def sample_fixed(data_a, data_b, sample_length):
    """sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    frames_total = len(data_a)
    frames_to_keep = frames_total // sample_length * sample_length

    start = np.random.randint(frames_total - frames_to_keep + 1)
    # print(f"Random crop from: {start}")
    end = start + frames_to_keep

    return data_a[start:end], data_b[start:end]





'''
description: 计算SI-SNR
param {*} preds
param {*} target
return {*}
'''
def SI_SDR(preds, target):
    EPS = 1e-8
    alpha = (np.sum(preds * target, axis=-1, keepdims=True) + EPS) / (np.sum(target ** 2, axis=-1, keepdims=True) + EPS)
    target_scaled = alpha * target
    noise = target_scaled - preds
    si_sdr_value = (np.sum(target_scaled ** 2, axis=-1) + EPS) / (np.sum(noise ** 2, axis=-1) + EPS)
    si_sdr_value = 10 * np.log10(si_sdr_value)
    return si_sdr_value

def SNR(preds, target):
    EPS = 1e-8
    # alpha = (np.sum(preds * target, axis=-1, keepdims=True) + EPS) / (np.sum(target ** 2, axis=-1, keepdims=True) + EPS)
    # target_scaled = alpha * target
    noise = target - preds
    si_sdr_value = (np.sum(target ** 2, axis=-1) + EPS) / (np.sum(noise ** 2, axis=-1) + EPS)
    si_sdr_value = 20 * np.log10(si_sdr_value)
    return si_sdr_value

if __name__ == '__main__':
    a = np.random.randn(1,1,48000)

    b = np.random.randn(1, 1,48000)

    c = LSD_NEW(a,b)
    print(c)
    d = SI_SDR(a,b)
    print(d)

