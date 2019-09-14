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
    def __init__(self):
        self.__seed = 18473

    def seed(self, seed):
        self.__seed = seed

    def output_seed(self):
        return self.__seed

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
                        csv_gen['img'] = cv2.imread(os.path.join(train_path, img)).tostring()
                    if ClassId == '4':
                        csv_gen['label'] = self.__mklabel(csv_gen['label']).tostring()
                        yield csv_gen
                        csv_gen = {'img': None, 'label': []}
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
            return tf.io.FixedLenFeature([], tf.string)
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
            tfr_path = os.path.join(dir_path, '%03d-of-%03d.tfrecord' % (num, num_shards))
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

    def _pseudo_random(self, a, b):
        """
        Linear congruential generator
        - https://en.wikipedia.org/wiki/Linear_congruential_generator
        """
        m = 2 ** 32
        seed = self.seed
        while True:
            nextseed = (a * seed + b) % m
            yield nextseed
            seed = nextseed

    def readtfrecorde(self, feature_dict, decode_raw, tfr_path, shuffle_buffer, num_valid, compression):
        random_gen = self._pseudo_random(214013, 2531011)
        files = tf.io.match_filenames_once(tfr_path)

        features = {}
        for key, _type in feature_dict.items():
            features[key] = self.__fix_len_feature(_type)

        self.dataset = tf.data.TFRecordDataset(files)
        self.dataset = self.dataset.map(lambda raw: tf.parse_single_example(raw, features=features))
        self.dataset = self.dataset.map(decode_raw)
        self.dataset = self.dataset.shuffle(shuffle_buffer, seed=next(random_gen))

        self.valid_dataset = self.dataset.take(num_valid)
        self.train_dataset = self.dataset.skip(num_valid)






