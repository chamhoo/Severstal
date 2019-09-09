import re
import os
import cv2
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from time import time
from functools import reduce
from tools.mask import rle2mask


class DataReader(object):
    def read_train(self, path, train_path, height, width, col=False, sep=','):
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
        targettime = 0
        totoaltime = time()

        csv_gen, numlen = {'img': None, 'label': []}, 0
        with open(path) as file:
            for line in file:
                line = re.split(sep, line.strip())
                img, ClassId, rle = line
                numlen += 1
                if (numlen == 1) & (not col):
                    self.col = line
                else:
                    csv_gen['label'].append(rle)
                    if ClassId == '1':
                        # csv_gen['img'] = bytes(img, encoding='utf-8')
                        csv_gen['img'] = cv2.imread(os.path.join(train_path, img)).tobytes()
                    if ClassId == '4':
                        now = time()
                        csv_gen['label'] = self.__mklabel(csv_gen['label']).tobytes()
                        # csv_gen['label'] = bytes(str(csv_gen['label']), encoding='utf-8')
                        targettime += (time() - now)
                        yield csv_gen
                        csv_gen = {'img': None, 'label': []}

        print(targettime)
        print(time() - totoaltime)

    def read_test(self, test_path):
        # test generator
        for path in os.listdir(test_path):
            if self.__isimg(path):
                yield {'img': cv2.imread(os.path.join(test_path, path))}

    def __isimg(self, path):
        return os.path.splitext(path)[1] in ['.png', '.jpg']

    def count(self, path):
        return reduce(lambda x, y: x+y, [self.__isimg(i) for i in os.listdir(path)])

    def __mklabel(self, rle):
        label_lst = [rle2mask(i, self.height, self.width) for i in rle]
        return np.concatenate(label_lst, axis=2)

    def __type_feature(self, _type, value):
        if _type == 'bytes':
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        if _type == 'int':
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        if _type == 'float':
            return tf.train.Feature(float64_list=tf.train.FloatList(value=[value]))

    def __fix_len_feature(self, _type):
        if _type == 'bytes':
            return tf.io.FixedLenFeature([], tf.strings)
        if _type == 'int':
            return tf.io.FixedLenFeature([], tf.int64)
        if _type == 'float':
            return tf.io.FixedLenFeature([], tf.int64)

    def __compression_tfr(self, compression_type='', c_level=None):
        """
        :param compression_type: 'GZIP', 'ZLIB' or ''
        """
        return tf.io.TFRecordOptions(compression_type=compression_type, compression_level=c_level)

    def write_tfr(self, data_generator, count, tfrpath, feature_dict, shards=1000,
                  compression=None, c_level=None):
        """

        :param data_generator:
        :param count:
        :param tfrpath:
        :param feature_dict: dict, 是用来记录feature内容的dict.
         结构为 {'key1': '_type1', 'key2': '_type2', ...} ,其中,
         key 必须与 data_generator 中的 key 对应, '_type' 来自 list
         ['int', 'float', 'bytes'].
        :param shards:
        :param compression:
        :return:
        """
        options = self.__compression_tfr(compression, c_level=c_level)
        # base on num_shards & count, build a slice list
        if shards <= 100:
            num_shards, step = int(shards), int(np.ceil(count/shards))
        else:
            num_shards, step = int(np.ceil(count/shards)), int(shards)

        # update dir
        dir_path = os.path.join('..', 'tmp', 'TFRecords', f'{tfrpath}')
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

        for num in range(num_shards):
            tfr_path = os.path.join(dir_path, '%03d-of-%03d.tfrecords' % (num, num_shards))
            writer = tf.io.TFRecordWriter(tfr_path, options=options)
            # write TFRecords file.
            try:
                for _ in tqdm(range(step)):
                    samples = next(data_generator)
                    # build feature
                    feature = {}
                    for key, _type in feature_dict.items():
                        feature[key] = self.__type_feature(_type, samples[key])

                    # build example
                    exmaple = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(exmaple.SerializeToString())

            # 如果全部数据迭代完成，利用 except 阻止抛出错误，并结束迭代。
            # If all data is iteratively completed, use the "except" to
            # prevent throwing errors and end the iteration.
            except StopIteration:
                pass

            finally:
                writer.close()

    def parser(self):
        pass

    def readtfrecorde(self, feature_dict, tfr_path, shuffle, compression, c_level):
        files = tf.train.match_filenames_once(tfr_path)
        self.dataset = tf.data.TFRecordDataset(files)


