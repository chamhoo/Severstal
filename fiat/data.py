"""
auther: leechh
"""
import os
import tensorflow as tf
from tqdm import trange
from math import ceil
from fiat.chunk import chunk
from fiat.os import mkdir


class TFR(object):
    def __init__(self, path, count, feature_dict, shards=1000, compression=None, c_level=None):
        """

        :param path:
        :param count:
        :param compression: 'GZIP', 'ZLIB' or ''
        :param c_level:
        """

        self.path = path
        self.count = count
        self.feature_dict = feature_dict
        self.shards = shards
        self.compression = compression
        self.c_level = c_level
        self.__seed = 18473

        if shards >= 100:
            self.shards = ceil(self.count / shards)

        count_lst = chunk(count, ceil(count / self.shards))
        self.countprefile = dict(zip(range(1, len(count_lst) + 1), count_lst))

    def seed(self, seed):
        self.__seed = seed

    def output_seed(self):
        return self.__seed

    @staticmethod
    def __type_feature(_type, value):
        if _type == 'bytes':
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        if _type == 'int':
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        if _type == 'float':
            return tf.train.Feature(float64_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def __fix_len_feature(_type):
        if _type == 'bytes':
            return tf.io.FixedLenFeature([], tf.string)
        if _type == 'int':
            return tf.io.FixedLenFeature([], tf.int64)
        if _type == 'float':
            return tf.io.FixedLenFeature([], tf.float32)

    def __options(self):
        return tf.io.TFRecordOptions(compression_type=self.compression, compression_level=self.c_level)

    def tfrecordname(self, idx):
        return os.path.join(self.path, '%03d-of-%03d.tfrecord' % (idx, self.shards))

    def write(self, data_generator, silence=False):
        """

        :param data_generator:
        :param feature_dict: dict, 是用来记录feature内容的dict.
         结构为 {'key1': '_type1', 'key2': '_type2', ...} ,其中,
         key 必须与 data_generator 中的 key 对应, '_type' 来自 list
         ['int', 'float', 'bytes'].
        :param shards:
        :return:
        """
        # base on num_shards & count, build a slice list

        # update dir
        mkdir(self.path)
        chunk_gen = chunk(self.count, ceil(self.count / self.shards))

        for idx, step in enumerate(chunk_gen):
            idx += 1
            tfrpath = self.tfrecordname(idx)
            writer = tf.io.TFRecordWriter(tfrpath, options=self.__options())
            self.countprefile[idx] = 0
            # write TFRecords file.
            try:
                if silence:
                    _range = range(step)
                else:
                    _range = trange(step)
                for _ in _range:
                    self.countprefile[idx] += 1
                    samples = next(data_generator)
                    # build feature
                    feature = {}
                    for key, _type in self.feature_dict.items():
                        feature[key] = TFR.__type_feature(_type, samples[key])

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

    def read(self, decode_raw, split=1, valid=0, shuffle_buffer=100, augment=None,
             buffer_size=None, num_parallel_reads=None, seed=None):
        """

        :param decode_raw:
        :param split:
        :param valid:
        :param shuffle_buffer:
        :param buffer_size:
        :param num_parallel_reads:
        :param seed:
        :return:
        """
        if seed is None:
            seed = self.__seed

        countprefile = self.countprefile

        def readfunc():
            datadict = {}
            idx_dict = {}
            count_dict = {}

            step = int(self.shards / split)
            idx_dict['valid'] = set([step*valid+i+1 for i in range(step)])
            idx_dict['train'] = set(range(1, self.shards+1)) - idx_dict['valid']
            for idx, val in idx_dict.items():
                files = tf.data.Dataset.list_files([self.tfrecordname(i) for i in val], shuffle=True, seed=seed)
                # features
                features = {}
                for key, _type in self.feature_dict.items():
                    features[key] = TFR.__fix_len_feature(_type)

                dataset = tf.data.TFRecordDataset(files,
                                                  compression_type=self.compression,
                                                  buffer_size=buffer_size,
                                                  num_parallel_reads=num_parallel_reads)
                dataset = dataset.map(lambda raw: tf.io.parse_single_example(raw, features=features))
                dataset = dataset.shuffle(shuffle_buffer, seed=seed)
                dataset = dataset.map(decode_raw)

                # DataAugment
                if (idx == 'train') & (augment is not None):
                    if type(augment) is list:
                        for subfunc in augment:
                            dataset = subfunc(dataset)
                    else:
                        dataset = augment(dataset)

                datadict[idx] = dataset
                count_dict[idx] = sum([countprefile[i] for i in val])

            return datadict, count_dict['train'], count_dict['valid']
        return readfunc
