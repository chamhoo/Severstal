import re
import numpy as np
import tensorflow as tf
from time import time
from tools.mask import rle2mask


class DataReader(object):
    def read_csv(self, path, height, width, col=False, sep=','):
        """
        read csv file to generator
        :param path: str, Path of the csv file.
        :param col: False(Bool) or list, if False, This mean that the col information is
         contained in the csv file. if not contained, You need to set col_name manually.
        :param sep: str, default ',' Delimiter to use.
        :return: generator
        """
        self.height = height
        self.width = width
        self.col = col
        csv_gen, self.count, numlen = [None, []], 0, 0
        with open(path) as file:
            for line in file:
                line = re.split(sep, line.strip())
                img, ClassId, rle = line
                numlen += 1
                if (numlen == 1) & (not col):
                    self.col = line
                else:
                    csv_gen[1].append(rle)
                    if ClassId == '1':
                        csv_gen[0] = img
                    if ClassId == '4':
                        yield self.__mklabel(csv_gen)
                        self.count += 1
                        csv_gen = [None, []]

    def __mklabel(self, csv_gen):
        label_lst = [rle2mask(i, self.height, self.width)[:, :, np.newaxis] for i in csv_gen[1]]
        return [csv_gen[0], np.concatenate(label_lst, axis=2)]

    def data2tfrecorde(self):
        pass

    def readtfrecorde(self):
        pass