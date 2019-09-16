"""
auther: leechh
"""
import os
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tools.data_gen import DataGen


class TFR(DataGen):
    def __init__(self):
        self.__seed = 18473

    def seed(self, seed):
        self.__seed = seed

    def output_seed(self):
        return self.__seed

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





