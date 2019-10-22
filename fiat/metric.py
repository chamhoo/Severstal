"""
auther: leechh

metric - large is better
"""
import tensorflow as tf


def dice():
    def metric(y_true, y_pred):
        smooth = 1e-8
        y_true = tf.keras.backend.flatten(y_true)
        y_pred = tf.keras.backend.flatten(y_pred)
        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return ((2 * intersection) + smooth) / (union + smooth)
    return metric


def mean_dice(axis=[1, 2]):
    def metric(y_true, y_pred):
        smooth = 1e-8
        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=axis)
        union = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis)
        return tf.reduce_mean(((2 * intersection) + smooth) / (union + smooth))
    return metric


def metricfromname(metric_name, y_true, y_pred):
    # y_*: [batch, height, width, num_class]
    # dice
    if metric_name == 'dice':
        return dice()(y_true, y_pred)

    if metric_name == 'mean_dice':
        return mean_dice()(y_true, y_pred)
    
    else:
        assert False, 'metric function ISNOT exist'
