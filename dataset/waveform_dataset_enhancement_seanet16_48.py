'''
Author: yehongcen && as_yhc@163.com
Date: 2024-04-23 15:36:41
LastEditors: redust as_yhc@163.com
LastEditTime: 2024-06-07 21:28:46
FilePath: \学位论文代码speechSR\dataset\waveform_dataset_enhancement_seanet16_48.py
Description: 推理dataset
Copyright (c) 2024 by yehongcen, All Rights Reserved. 
'''
import os
from torch.utils.data import Dataset
import librosa
import random
from util.utils import sample_fixed_length_data_aligned,normalize_data,sample_fixed,normalize_data_enhance
from scipy.signal.fir_filter_design import firwin
from scipy.signal import dlti
from scipy.signal.filter_design import cheby1
from scipy.signal.signaltools import filtfilt, lfilter
from scipy.signal.signaltools import resample_poly
from scipy.signal._upfirdn import upfirdn
import operator
import numpy as np


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


class WaveformDataset(Dataset):
    def __init__(self, dataset, limit=None, offset=0, sample_length=16384):
        """Construct dataset for enhancement.
        Args:
            dataset (str): *.txt. The path of the dataset list file. See "Notes."
            limit (int): Return at most limit files in the list. If None, all files are returned.
            offset (int): Return files starting at an offset within the list. Use negative values to offset from the end of the list.

        Notes:
            dataset list file：
            <noisy_1_path>
            <noisy_2_path>
            ...
            <noisy_n_path>

            e.g.
            /enhancement/noisy/a.wav
            /enhancement/noisy/b.wav
            ...

        Return:
            (mixture signals, filename)
        """
        super(WaveformDataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.down_rate = 3
        self.downsampling = 'cheby'
        self.sr = 48000

    def __len__(self):
        return self.length
    '''
    description: 低通
    param {*} self
    param {*} sig
    return {*}
    '''
    def lowpass(self, sig):
        low_sr = 16000

        if self.downsampling == 'cheby':
            sig = decimate(sig, self.down_rate)
            sig = librosa.resample(sig, orig_sr=low_sr, target_sr=self.sr)
        else:
            sig = librosa.resample(sig, orig_sr=self.sr, target_sr=low_sr, res_type=self.downsampling)
            sig = librosa.resample(sig, orig_sr=low_sr, target_sr=self.sr)
        return sig


    def __getitem__(self, item):
        mixture_path = self.dataset_list[item]
        name = os.path.splitext(os.path.basename(mixture_path))[0]

        mixture, _ = librosa.load(os.path.abspath(os.path.expanduser(mixture_path)), sr=48000)
        # mixture, max_bone = normalize_data_enhance(mixture)
        low_sig = self.lowpass(mixture) # 降采样
        if len(mixture) != len(low_sig):
            low_sig = pad(low_sig, len(mixture))
        return low_sig.reshape(1, -1), name
