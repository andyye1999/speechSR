import numpy as np
from pydub import AudioSegment
import soundfile as sf
import os
import matplotlib.pyplot as plt
import sys


def snr(x, y):    
    return 10*np.log10((np.abs(x-y)**2).mean() / (np.abs(x)**2).mean())

def estimate_int_delay(x, y):
    Cyx = np.correlate(x, y, 'full')
    n = y.shape[0] + x.shape[0] - 1
    lags = np.arange(-n//2, n//2, 1)
    Cm = np.abs(Cyx)
    imax = np.where(Cm == Cm.max())[0]
    int_delay = int(lags[imax]+1)
    return int_delay

def fix_int_delay(x, y, d):
    if d > 0:
        x = x[d:]
        y = y[:-d]
    elif d < 0:    
        x = x[:d]
        y = y[-d:]
    else: #d == 0
        pass
    return x, y

def align_signals(x, y, debug=False):
    if x.shape[0] != y.shape[0]:
        n = min(x.shape[0], y.shape[0])
        x = x[:n]
        y = y[:n]
    
    if debug:
        plt.plot(x[:3000])
        plt.plot(y[:3000])
        plt.show()

    d = estimate_int_delay(x, y)    
    x, y = fix_int_delay(x, y, d)    
    return x, y

if __name__ == "__main__":    
    fsize_wav = os.path.getsize("arctic_a0001.wav")
    fsize_compressed = os.path.getsize("arctic_a0001.aac")
    print(fsize_wav, fsize_compressed, fsize_compressed/fsize_wav*100)
    samples, fs = sf.read("arctic_a0001.wav")
    audio = AudioSegment.from_file("arctic_a0001.aac", "aac") 
    samples_comp = np.array(audio.get_array_of_samples())
    samples_comp = (samples_comp/2**15).astype(np.float32)
    print(samples.shape, samples_comp.shape)

    samples, samples_comp = align_signals(samples, samples_comp, True)
    print(samples.shape, samples_comp.shape)

    samples, samples_comp = align_signals(samples, samples_comp, True)
    print(samples.shape, samples_comp.shape)

    print(snr(samples, samples_comp))