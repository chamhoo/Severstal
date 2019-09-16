"""
auther: leechh
"""
import os
import re
import cv2
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

    def seg_train_gen(self, path, train_path, height, width, col=False, sep=','):
        """
        read csv file to generator
        :param path: str, Path of the csv file.
        :param col: False(Bool) or list, if False, This mean that the col information is
         contained in the csv file. if not contained, You need to set col_name manually.
        :param sep: str, default ',' Delimiter to use.
        :return: generator
        """
        col = col
        totoaltime = time()

        csv_gen, numlen = {'img': None, 'label': []}, 0
        with open(path) as file:
            for line in file:
                line = re.split(sep, line.strip())
                img, ClassId, rle = line
                numlen += 1
                if (numlen == 1) & (not col):
                    col = line
                else:
                    csv_gen['label'].append(rle)
                    if ClassId == '1':
                        csv_gen['img'] = cv2.imread(os.path.join(train_path, img)).astype('uint8').tostring()
                    if ClassId == '4':
                        csv_gen['label'] = self.__mklabel(csv_gen['label'], height, width).astype('bool').tostring()
                        yield csv_gen
                        csv_gen = {'img': None, 'label': []}
        print(time() - totoaltime)

    def read_test(self, test_path):
        # test generator
        for path in os.listdir(test_path):
            if self.__isimg(path):
                yield {'img': cv2.imread(os.path.join(test_path, path)).astype('uint8').tostring()}

    def count(self, path):
        return reduce(lambda x, y: x+y, [self.__isimg(i) for i in os.listdir(path)])