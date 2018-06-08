# encoding utf-8


# Created:    on June 7, 2018 20:10
# @Author:    xxoospring

r"""spectrum data generate
########################
All wav file names should be in this format:
# wav files name:
#     0_*.wav:
#         0 represent class label
########################

    --wave_path: wav files store path
    --spec_path: spectrum txt data store path
    --isCrop: default True
            True:   for long wav file, suggest to crop it into small pieces file.Such as a sound record of
                a motor
            False:  generate spectrum data for the whole wav file
    --pace:crop long wav file per 'pace', default 1000ms(s).If isCrop=False, this param is invalid.
    --used_max_area: default: True
            True:   get_max_pos will be called to get the sum maximum area in a spectrum data 
    --area_size: square window size for searching  maximum area, default 100.If used_max_area=False,
                this param is invalid
Example usage:
    python spec_data_gen.py --wave_path=YOU_DATA_PATH --spec_path=SPECTRUM_STORE_PATH
"""
from warnings import warn

import tensorflow as tf
import os
import sys
sys.path.append('../pub/')
sys.path.append('./')
from file import file_filter
from spectrum import wav_spectrum


flags = tf.app.flags
flags.DEFINE_string('wave_path', ' ', 'wav data files path')
flags.DEFINE_string('spec_path', ' ', 'spectrum data output path')
flags.DEFINE_boolean('isCrop', True, 'crop long wave ')
flags.DEFINE_integer('pace', 1000, 'long wav file crop pace')
flags.DEFINE_boolean('used_max_area', True, 'select the maxmiu sum area')
flags.DEFINE_integer('area_size', 100, 'area size')
FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.wave_path):
        raise ValueError('wave path not found!')
    if not os.path.exists(FLAGS.spec_path):
        raise ValueError('store path not found!')
    files = file_filter(FLAGS.wave_path, 'wav')
    illegal_name_format = []
    for f in files:
        print(f)
        if not f.split('_')[0].isdigit():
            illegal_name_format.append(f)
            continue
        wav_spectrum(FLAGS.wave_path+f,
                     FLAGS.spec_path,
                     iscrop=FLAGS.isCrop,
                     pace_ms=FLAGS.pace,
                     used_max_area=FLAGS.used_max_area,
                     s_size=FLAGS.area_size
                     )
    for name in illegal_name_format:
        warn(name + ' foramt is illegal.')

if __name__ == '__main__':
    tf.app.run()