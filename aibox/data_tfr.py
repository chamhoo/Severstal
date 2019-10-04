"""
auther: leechh
"""
import os
import shutil
import numpy as np
import tensorflow as tf
from tqdm import trange
from math import ceil
from aibox.data_gen import DataGen
from aibox.chunk import chunk


class TFR(DataGen):
    def __init__(self):
        self.__seed = 18473
        self.random_gen = self._pseudo_random(214013, 2531011, self.__seed)

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
            return tf.io.FixedLenFeature([], tf.float32)

    def __compression_tfr(self, compression_type='', c_level=None):
        """
        :param compression_type: 'GZIP', 'ZLIB' or ''
        """
        return tf.io.TFRecordOptions(compression_type=compression_type, compression_level=c_level)

    def mkdir(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def write_tfr(self, data_generator, tfrpath, train_path, feature_dict, shards=1000,
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
        count = self.count(train_path)
        options = self.__compression_tfr(compression, c_level=c_level)
        # base on num_shards & count, build a slice list
        if shards <= 100:
            shards = ceil(count / shards)

        # update dir
        self.mkdir(tfrpath)
        chunk_gen = chunk(count, shards)
        for num, step in enumerate(chunk(count, shards)):
            tfr_path = os.path.join(tfrpath, '%03d-of-%03d.tfrecord' % (num, chunk_gen.__len__()))
            writer = tf.io.TFRecordWriter(tfr_path, options=options)
            # write TFRecords file.
            try:
                for _ in trange(step):
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
                break

            finally:
                writer.close()

    def _pseudo_random(self, a, b, seed):
        """
        Linear congruential generator
        - https://en.wikipedia.org/wiki/Linear_congruential_generator
        """
        m = 2 ** 32
        while True:
            nextseed = (a * seed + b) % m
            yield nextseed
            seed = nextseed

    def readtfrecorde(self, feature_dict, decode_raw, tfr_path,
                      shuffle_buffer, compression, buffer_size=None, num_parallel_reads=None):
        files = tf.data.Dataset.list_files(tfr_path, shuffle=True, seed=self.__seed)
        # files = tf.io.match_filenames_once(tfr_path)

        features = {}
        for key, _type in feature_dict.items():
            features[key] = self.__fix_len_feature(_type)

        dataset = tf.data.TFRecordDataset(files,
                                          compression_type=compression,
                                          buffer_size=buffer_size,
                                          num_parallel_reads=num_parallel_reads)
        dataset = dataset.map(lambda raw: tf.io.parse_single_example(raw, features=features))
        dataset = dataset.shuffle(shuffle_buffer, seed=self.__seed)
        dataset = dataset.map(decode_raw)
        return dataset

    def readtrain(self, rt_params, train_path, epoch, batch_size, reshape=None, reshape_method=None):
        """
        AREA = 3
        BICUBIC = 2
        BILINEAR = 0
        NEAREST_NEIGHBOR = 1
        """
        # parameter
        self.epoch = epoch
        self.batch_size = batch_size
        self.reshape = reshape
        self.reshape_method  = reshape_method

        # read tfrecord & resize
        dataset = self.readtfrecorde(**rt_params)
        self.traincount = self.count(train_path)
        self.origion_shape = self.mkshape(train_path)

        # train & valid dataset
        dataset = dataset.batch(self.batch_size).repeat(self.epoch)
        self.iterator = dataset.make_initializable_iterator()
        self.img_origin, self.label = self.iterator.get_next()
        self.img = tf.image.resize(self.img_origin, size=reshape, method=reshape_method)




