"""
auther: leechh
"""
import tensorflow as tf
from math import sqrt


class Variable(object):
    @staticmethod
    def weight_variable(shape, type_name, weight_param=None, name='weight'):
        """

        :param shape:
        :param weight_param:
            truncated_normal - stddev
            normal - mean, stddev
            uniform - minval, maxval
        :param type_name:
        :param name:
        :return:
        """
        # truncated_normal:
        if type_name == 'truncated_normal':
            return tf.Variable(tf.truncated_normal(shape, **weight_param), name=name)
        # normal:
        elif type_name == 'normal':
            return tf.Variable(tf.random.normal(shape, **weight_param), name=name)
        # uniform
        elif type_name == 'uniform':
            return tf.Variable(tf.random.uniform(shape, **weight_param), name=name)
        # zeros
        elif type_name == 'zeros':
            return tf.Variable(tf.zeros(shape), name=name)
        # ones
        elif type_name == 'ones':
            return tf.Variable(tf.ones(shape), name=name)
        # assert
        else:
            assert False, 'initial name ISNOT exist'

    @staticmethod
    def bias_variable(value, shape, name='bias'):
        return tf.Variable(tf.constant(value=value, shape=shape, dtype='float32'), name=name)

    @staticmethod
    def cal_std(filter_size, channel):
        return sqrt(2 / (filter_size**2 * channel))
