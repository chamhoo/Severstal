"""
auther: leechh
"""

import os
import re
import cv2
import copy
import numpy as np
import tensorflow as tf
from time import time
from fiat.mask import rle2mask
from fiat.os import mkshape


feature_dict = {
    'img': 'bytes',
    'label': 'bytes',
    'height': 'int',
    'width': 'int',
    'channels': 'int',
    'n_class': 'int'
}


def __mklabel(rle, height, width, nclass):
    label_lst = [rle2mask(i, height, width) for i in rle]
    label_arr = np.concatenate(label_lst, axis=2)
    if nclass == 5:
        background = (1 - np.sum(label_arr, axis=2, dtype='bool'))[..., np.newaxis]
        label_arr = np.concatenate([label_arr, background], axis=2)
    return label_arr


def seg_train_gen(csv_path, train_path, col=False, sep=',', nclass=5):
    """
    read csv file to generator
    :param csv_path: str, Path of the csv file.
    :param train_path:
    :param col: False(Bool) or list, if False, This mean that the col information is
     contained in the csv file. if not contained, You need to set col_name manually.
    :param sep: str, default ',' Delimiter to use.
    :return: generator
    """
    numlen = 0
    totoaltime = time()

    height, width, _ = mkshape(train_path)

    csv_gen_init = {'img': None, 'label': [],
                    'height': height, 'width': width,
                    'channels': 3, 'n_class': nclass}

    csv_gen = copy.deepcopy(csv_gen_init)
    with open(csv_path) as file:
        for line in file:
            line = re.split(sep, line.strip())
            img, ClassId, rle = line
            numlen += 1
            if (numlen == 1) & (not col):
                col = line
            else:
                csv_gen['label'].append(rle)
                if ClassId == '1':
                    csv_gen['img'] = cv2.imread(os.path.join(train_path, img)).astype('uint8').tobytes()
                if ClassId == '4':
                    csv_gen['label'] = __mklabel(csv_gen['label'], height, width, nclass).astype('uint8').tobytes()
                    yield csv_gen
                    csv_gen = copy.deepcopy(csv_gen_init)

    print(time() - totoaltime)


def decode_img_seg(raw):
    img = tf.decode_raw(raw['img'], tf.uint8)
    label = tf.decode_raw(raw['label'], tf.uint8)

    # convert image dtype
    img = tf.cast(img, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.float32)

    # set_shape
    img = tf.reshape(img, [raw['height'], raw['width'], raw['channels']])
    label = tf.reshape(label, [raw['height'], raw['width'], raw['n_class']])
    return img, label