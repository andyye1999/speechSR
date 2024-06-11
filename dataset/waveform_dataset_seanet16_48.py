'''
Author: yehongcen && as_yhc@163.com
Date: 2024-04-23 15:36:41
LastEditors: redust as_yhc@163.com
LastEditTime: 2024-06-07 21:25:50
FilePath: \学位论文代码speechSR\dataset\waveform_dataset_seanet16_48.py
Description: 训练用dataset
Copyright (c) 2024 by yehongcen, All Rights Reserved. 
'''
import os
import random
import librosa
from torch.utils import data
import numpy as np
from util.utils import sample_fixed_length_data_aligned,normalize_data,sample_fixed_length_data_alignedss
import operator
from scipy.signal.fir_filter_design import firwin
from scipy.signal import dlti
from scipy.signal.filter_design import cheby1
from scipy.signal.signaltools import filtfilt, lfilter
from scipy.signal.signaltools import resample_poly
from scipy.signal._upfirdn import upfirdn

# 滤波器选择
def decimate(x, q, ripple=0.05, n=None, ftype='iir', axis=-1, zero_phase=True):
    x = np.asarray(x)
    q = operator.index(q)

    if n is not None:
        n = operator.index(n)

    if ftype == 'fir':
        if n is None:
            half_len = 10 * q  # reasonable cutoff for our sinc-like function
            n = 2 * half_len
        b, a = firwin(n + 1, 1. / q, window='hamming'), 1.
    elif ftype == 'iir':
        if n is None:
            n = 8
        system = dlti(*cheby1(n, ripple, 0.8 / q))
        b, a = system.num, system.den
    elif isinstance(ftype, dlti):
        system = ftype._as_tf()  # Avoids copying if already in TF form
        b, a = system.num, system.den
    else:
        raise ValueError('invalid ftype')

    result_type = x.dtype
    if result_type.kind in 'bui':
        result_type = np.float64
    b = np.asarray(b, dtype=result_type)
    a = np.asarray(a, dtype=result_type)

    sl = [slice(None)] * x.ndim
    a = np.asarray(a)

    if a.size == 1:  # FIR case
        b = b / a
        if zero_phase:
            y = resample_poly(x, 1, q, axis=axis, window=b)
        else:
            n_out = x.shape[axis] // q + bool(x.shape[axis] % q)
            y = upfirdn(b, x, up=1, down=q, axis=axis)
            sl[axis] = slice(None, n_out, None)

    else:
        if zero_phase:
            y = filtfilt(b, a, x, axis=axis)
        else:
            y = lfilter(b, a, x, axis=axis)
        sl[axis] = slice(None, None, q)

    return y[tuple(sl)]


def pad(sig, length):
    if len(sig) < length:
        pad = length - len(sig)
        sig = np.hstack((sig, np.zeros(pad) + 0.1))
    else:
        start = random.randint(0, len(sig) - length)
        sig = sig[start:start + length]
    return sig


class Dataset(data.Dataset):
    def __init__(self,
                 dataset,
                 limit=None,
                 offset=0,
                 sample_length=16384,
                 mode="train"):
        """Construct dataset for training and validation.
        Args:
            dataset (str): *.txt, the path of the dataset list file. See "Notes."
            limit (int): Return at most limit files in the list. If None, all files are returned.
            offset (int): Return files starting at an offset within the list. Use negative values to offset from the end of the list.
            sample_length(int): The model only supports fixed-length input. Use sample_length to specify the feature size of the input.
            mode(str): If mode is "train", return fixed-length signals. If mode is "validation", return original-length signals.

        Notes:
            dataset list file：
            <noisy_1_path><space><clean_1_path>
            <noisy_2_path><space><clean_2_path>
            ...
            <noisy_n_path><space><clean_n_path>

            e.g.
            /train/noisy/a.wav /train/clean/a.wav
            /train/noisy/b.wav /train/clean/b.wav
            ...

        Return:
            (mixture signals, clean signals, filename)
        """
        super(Dataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        assert mode in ("train", "validation"), "Mode must be one of 'train' or 'validation'."

        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.mode = mode
        self.down_rate = 3 # xiacaiyangbeishu
        self.downsampling = 'cheby'
        self.sr = 48000

    def __len__(self):
        return self.length




    '''
    description: 低通滤波器
    param {*} self
    param {*} sig
    return {*}
    '''
    def lowpass(self, sig):
        low_sr = 16000

        if self.downsampling == 'cheby':
            sig = decimate(sig, self.down_rate)
            sig = librosa.resample(sig, orig_sr=low_sr, target_sr=self.sr) # 重采样
        else:
            sig = librosa.resample(sig, orig_sr=self.sr, target_sr=low_sr, res_type=self.downsampling)
            sig = librosa.resample(sig, orig_sr=low_sr, target_sr=self.sr)
        return sig

    def __getitem__(self, item):
        mixture_path, clean_path = self.dataset_list[item].split(" ")
        filename = os.path.splitext(os.path.basename(mixture_path))[0]
        mixture, _ = librosa.load(os.path.abspath(os.path.expanduser(mixture_path)), sr=48000)
        # clean, _ = librosa.load(os.path.abspath(os.path.expanduser(clean_path)), sr=8000)
        # mixture, max_bone = normalize_data_ss(mixture)

        if self.mode == "train": # 训练阶段
            # The input of model should be fixed-length in the training.
            mixture = sample_fixed_length_data_alignedss(mixture, self.sample_length) # 随机选取固定长度音频
            low_sig = self.lowpass(mixture)
            if len(mixture) != len(low_sig):
                low_sig = pad(low_sig, len(mixture))
            return low_sig.reshape(1, -1), mixture.reshape(1, -1), filename
        else: # 验证阶段
            low_sig = self.lowpass(mixture)
            if len(mixture) != len(low_sig):
                low_sig = pad(low_sig, len(mixture))
            return low_sig.reshape(1, -1), mixture.reshape(1, -1), filename
