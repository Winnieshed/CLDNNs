# encoding utf-8


# Created:    on June 7, 2018 20:10
# @Author:    xxoospring

r"""将wav文件转化为对应的语谱数据，保存为txt
1. swap:
    fft计算后的数据和ndarray维度不对应，低频在上面，高频在下面。倒置过来。
2. get_spectrum：
    将传进来的数据做fft,默认的窗口窗是
3.wave_spectrum：
    将长的wav文件切分成一定间隔的语谱数据
4.show:
    调试用
"""
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hann
import os


# swap an array
def swap(arr):  # only one dim
    l = len(arr)
    ret = np.zeros((l, 1))
    for i in range(l):
        ret[i] = arr[-i-1]
    return ret


# get 'wave_data' spectrum
def get_spectrum(wave_data, framerate, window_length_ms, window_shift_times):
    wav_length = len(wave_data)
    window_length = framerate * window_length_ms / 1000
    window_shift = int(window_length * window_shift_times)
    nframe = int(np.ceil((wav_length - window_length+1) / window_shift))
    freq_num = int(window_length/2)
    spec = np.zeros((freq_num, nframe))
    for i in range(nframe):
        start = int(i * window_shift)
        end = int(start + window_length)
        fft_slice = wave_data[start:end]
        fft_slice = fft_slice.astype('float64')
        fft_slice *= hann(end-start, sym=False)
        w_fft = np.abs(np.fft.fft(fft_slice))
        w_fft = np.clip(w_fft, a_min=1e-16, a_max=None)
        freq = w_fft[:int(window_length/2)]
        freq = swap(freq).reshape(freq_num)
        spec[:, i] = freq
    return spec


# My GPU memory is not sufficent for my training set.
# You can use this func to crop the spectrum into a given size array
# The maxium sum area is the most feature representativeness area
# Return the position
def get_max_pos(data, size=(100, 100), mini_crop=False):
    row, col = np.shape(data)
    if row < size[0] or col < size[1]:
        if not mini_crop:
            raise ValueError('Crop size is larger than data.Data Shape:%s' % (data.shape))
        else:
            c = min((row, col))
            size = (c, c)
    max_sum = 0.0
    pos = [(), ()]
    for x in range(col):
        if x+size[0] > col:
            break
        for y in range(row):
            if y+size[1] > row:
                break
            area_sum = np.sum(data[y:y+size[1], x:x+size[0]])
            if max_sum < area_sum:
                max_sum = area_sum
                pos[0] = (y, x)     # top left
                pos[1] = (y+size[1], x+size[0])     # bottom right
    return pos, size[0]


def wav_spectrum(
        filename,           # wave file
        store_path,         # spectrum store path
        iscrop=False,       # crop long wave file or not
        pace_ms=1000,       # crop pace, 1s default
        pace_shift=0.5,     # pace shift
        used_max_area=True,
        s_size=100
        ):
    wav_file = wave.open(filename, 'r')
    params = wav_file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = wav_file.readframes(wav_length)
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_len = len(wave_data)
    name = filename.split('/')[-1].split('.')[0]
    if not iscrop:
        spec = get_spectrum(wave_data, framerate, 10, 0.5)
        if used_max_area:
            pos, _ = get_max_pos(spec, mini_crop=True)
            spec_out = spec[pos[0][0]:pos[1][0], pos[0][1]:pos[1][1]]
            np.savetxt(store_path + name + '_' + '.txt', spec_out)
            # show(spec_out)
        else:
            np.savetxt(store_path+name +'.txt', spec)
        print(name +'.txt')
    else:
        pace_len = int(framerate * pace_ms / 1000)
        shift_len = pace_len * pace_shift
        crop_num = int(np.ceil((wave_len - pace_len + 1) / shift_len))
        for i in range(crop_num):
            start_pos = int(i*shift_len)
            end_pos = int(start_pos+pace_len)
            spec = get_spectrum(wave_data[start_pos:end_pos], framerate, 10, 0.5)
            if used_max_area:
                pos, _ = get_max_pos(spec, size=(s_size, s_size),)
                spec_out = spec[pos[0][0]:pos[1][0], pos[0][1]:pos[1][1]]
                np.savetxt(store_path+name + '_' + '%s.txt' % i, spec_out)
            else:
                np.savetxt(store_path+name + '_' + '%s.txt' % i, spec)
            print(name + '_' + '%s.txt' % i)


def show(spec):
    plt.subplot(111)
    plt.imshow(spec)
    plt.show()

if __name__ == '__main__':
    # wav_spectrum(
    #     '../data/test_data/0_0.wav',
    #     './',
    #     iscrop=True,
    #     # used_max_area=False
    #     # s_size=100
    # )
    pass