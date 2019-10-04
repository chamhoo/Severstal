"""
auther: leechh
"""
import os
import re
import cv2
import copy
import numpy as np
import tensorflow as tf
from glob import glob, iglob
from time import time
from functools import reduce
from aibox.mask import rle2mask


class DataGen(object):

    def __mklabel(self, rle, height, width):
        label_lst = [rle2mask(i, height, width) for i in rle]
        label_arr = np.concatenate(label_lst, axis=2)
        background = (1 - np.sum(label_arr, axis=2, dtype='bool'))[..., np.newaxis]
        return np.concatenate([label_arr, background], axis=2)

    # def __isimg(self, path):
    #     return os.path.splitext(path)[1] in ['.png', '.jpg']

    def seg_train_gen(self, csv_path, train_path, col=False, sep=',', n_class=5):
        """
        read csv file to generator
        :param csv_path: str, Path of the csv file.
        :param train_path:
        :param col: False(Bool) or list, if False, This mean that the col information is
         contained in the csv file. if not contained, You need to set col_name manually.
        :param sep: str, default ',' Delimiter to use.
        :return: generator
        """
        col = col
        numlen = 0
        totoaltime = time()

        height, width, _ = self.mkshape(train_path)

        csv_gen_init = {'img': None, 'label': [],
                        'height': height, 'width': width,
                        'channels': 3, 'n_class': n_class}

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
                        csv_gen['label'] = self.__mklabel(csv_gen['label'], height, width).astype('uint8').tobytes()
                        yield csv_gen
                        csv_gen = copy.deepcopy(csv_gen_init)

        print(time() - totoaltime)

    def count(self, path):
        return len(glob(os.path.join(path, '*.jpg')))

    def mkshape(self, path):
        return cv2.imread(glob(os.path.join(path, '*.jpg'))[0]).shape