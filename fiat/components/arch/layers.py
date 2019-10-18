"""
auther: leechh
"""
import tensorflow as tf


class Layers(object):
    @staticmethod
    def conv2d(x, w, b, padding='SAME', name='conv2d', rate=0.25, strides=1):
        with tf.name_scope(name):
            conv = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
            conv = tf.nn.bias_add(conv, b)
            return tf.nn.dropout(conv, rate=rate)

    @staticmethod
    def conv2d_batch_norm(x, w, b, padding='SAME', name='conv2d_batch_norm', rate=0.25, strides=1):
        with tf.name_scope(name):
            conv = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
            conv = tf.nn.bias_add(conv, b)
            conv = tf.layers.batch_normalization(conv)
            return tf.nn.dropout(conv, rate=rate)

    @staticmethod
    def deconv2d(x, w, b, strides, f=2, padding='SAME', name='deconv2d'):
        with tf.name_scope(name):
            x_shape = tf.shape(x)
            output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // f])
            deconv = tf.nn.conv2d_transpose(
                x, w,
                output_shape=output_shape,
                strides=[1, strides, strides, 1],
                padding=padding,
                name='upconv')
            return tf.nn.bias_add(deconv, b)

    @staticmethod
    def deconv2d_batch_norm(x, w, b, strides, padding='SAME', name='deconv2d_batch_norm'):
        deconv = Layers.deconv2d(x, w, b, strides, padding=padding, name=name)
        return tf.layers.batch_normalization(deconv)

    @staticmethod
    def maxpooling(x, n, s=None, padding='SAME'):
        if s is None:
            s = n
        return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, s, s, 1], padding=padding)

    @staticmethod
    def avgpooling(x, n, s=None, padding='SAME'):
        if s is None:
            s = n
        return tf.nn.avg_pool(x, ksize=[1, n, n, 1], strides=[1, s, s, 1], padding=padding)

    @staticmethod
    def concat(x_down, x_up):
        x_down_shape = tf.shape(x_down)   # (batch, down_height, down_width, down_features)
        x_up_shape = tf.shape(x_up)       # (batch, up_height, up_width, up_features)
        x_down_slice = tf.slice(
            x_down,
            begin=[0, (x_down_shape[1] - x_up_shape[1]) // 2, (x_down_shape[2] - x_up_shape[2]) // 2, 0],
            size=[-1, x_up_shape[1], x_up_shape[2], -1])
        # (batch, up_height, up_width, up_features + down_features)
        return tf.concat([x_down_slice, x_up], axis=3)
