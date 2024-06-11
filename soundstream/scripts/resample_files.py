import os
import glob
import librosa
import scipy.io.wavfile as wavfile
import multiprocessing


def process_file(f, fs_trg=8000):
    ff = f.split('/')
    ff[-4] = 'ARCTIC8k'
    ff = '/'.join(ff)
    dirname = os.path.dirname(ff)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # if os.path.isfile(ff):
    #     return True
    if os.path.getsize(f) < 100:
        return False
    x, fs = librosa.core.load(f, sr=None)
    if x.shape[0] == 0:
        return False
    if fs != fs_trg:
        x = librosa.core.resample(x, orig_sr=fs, target_sr=fs_trg)
    else:
        pass
    wavfile.write(ff, rate=fs_trg, data=x)
    return True


def resample_mp(root, root_dst, fs_trg):
    fnames = glob.glob(root + '/*/*/*.wav', recursive=True)
    print("found ", len(fnames))
    if not os.path.isdir(root_dst):
        os.mkdir(root_dst)
    p = multiprocessing.Pool()
    for i, f in enumerate(fnames):
        res = p.apply_async(process_file, [f, fs_trg])
        # res = process_file(f, fs_trg)        
    p.close()
    p.join()


if __name__ == '__main__':
    fs_trg = 8000
    root = '/media/avi/8E56B6E056B6C86B/datasets/ARCTIC'
    root_dst = '/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k'
    resample_mp(root, root_dst, fs_trg)
    print("DONE")