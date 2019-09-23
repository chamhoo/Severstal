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
from functools import reduce
from tools.mask import rle2mask


class DataGen(object):

    def __mklabel(self, rle, height, width):
        label_lst = [rle2mask(i, height, width) for i in rle]
        return np.concatenate(label_lst, axis=2)

    def __isimg(self, path):
        return os.path.splitext(path)[1] in ['.png', '.jpg']

    def seg_train_gen(self, csv_path, train_path, height, width, col=False, sep=','):
        """
        read csv file to generator
        :param csv_path: str, Path of the csv file.
        :param col: False(Bool) or list, if False, This mean that the col information is
         contained in the csv file. if not contained, You need to set col_name manually.
        :param sep: str, default ',' Delimiter to use.
        :return: generator
        """
        col = col
        numlen = 0
        totoaltime = time()

        csv_gen_init = {'img': None, 'label': [],
                        'height': height, 'width': width,
                        'channels': 3, 'n_class': 4}

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

    def read_test(self, test_path, height, width,):
        # test generator
        for path in os.listdir(test_path):
            if self.__isimg(path):
                yield {'img': cv2.imread(os.path.join(test_path, path)).astype('uint8').tobytes(),
                       'height': height,
                       'width': width,
                       'channels': 3}

    def count(self, path):
        return reduce(lambda x, y: x+y, [self.__isimg(i) for i in os.listdir(path)])