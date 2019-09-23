"""
auther: leechh
"""
import tensorflow as tf


class Layers(object):

    def weight_variable(self, shape, type_name, weight_param=None, name='weight'):
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

    def bias_variable(self, value, shape, name='bias'):
        return tf.Variable(tf.constant(value=value, shape=shape, dtype='float32'), name=name)

    def conv2d(self, x, w, b, padding='SAME', name='conv2d', rate=0.25):
        with tf.name_scope(name):
            conv = tf.nn.conv2d(x, w, strides=1, padding=padding)
            conv = tf.nn.bias_add(conv, b)
            return tf.nn.dropout(conv, rate=rate)

    def deconv2d(self, x, w, b, strids, padding='SAME', name='upconv2d'):
        with tf.name_scope(name):
            x_shape = tf.shape(x)
            output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
            upconv = tf.nn.conv2d_transpose(
                x, w,
                output_shape=output_shape,
                strides=[1, strids, strids, 1],
                padding=padding,
                name='upconv')
            return tf.nn.bias_add(upconv, b)

    def maxpooling(self, x, n, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding=padding)

    def concat(self, x_down, x_up):
        x_down_shape = tf.shape(x_down)   # (batch, down_height, down_width, down_features)
        x_up_shape = tf.shape(x_up)       # (batch, up_height, up_width, up_features)
        x_down_slice = tf.slice(
            x_down,
            begin=[0, (x_down_shape[1] - x_up_shape[1]) // 2, (x_down_shape[2] - x_up_shape[2]) // 2, 0],
            size=[-1, x_up_shape[1], x_up_shape[2], -1])
        # (batch, up_height, up_width, up_features + down_features)
        return tf.concat([x_down_slice, x_up], axis=3)


class ModelComponent(object):

    def optimizier(self, optimizier_name, learning_rate, loss):
        # Adam
        if optimizier_name == 'adam':
            train_opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8
            ).minimize(loss)
        # adagrad
        elif optimizier_name == 'adagrad':
            train_opt = tf.train.AdagradOptimizer(
                learning_rate=learning_rate,
                initial_accumulator_value=1e-8
            ).minimize(loss)
        # gd
        elif optimizier_name == 'gd':
            train_opt = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate,
            ).minimize(loss)
        # momentun
        elif optimizier_name == 'momentun':
            train_opt = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=0.99
            ).minimize(loss)

        else:
            assert False, 'optimizer name ISNOT exist'

        return train_opt

    def metric_func(self, metric_name, y_true, y_pred):
        # y_*: [batch, height, width, num_class]
        # dice
        if metric_name == 'dice':
            smooth = 1e-8
            # y_true = tf.nn.softmax(y_true)
            # y_pred = tf.nn.softmax(y_pred)
            intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
            y_true_sum = tf.reduce_sum(y_true)
            y_pred_sum = tf.reduce_sum(y_pred)
            y_true_count = tf.count_nonzero(y_true)
            y_pred_count = tf.count_nonzero(y_pred)
            union = y_true_sum + y_pred_sum
            return [
                1 - (((2 * intersection) + smooth) / (union + smooth)),
                intersection,
                y_true_sum, y_pred_sum,
                y_true_count, y_pred_count,
                y_true, y_pred]

        else:
            assert False, 'metric function ISNOT exist'
