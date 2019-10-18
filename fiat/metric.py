"""
auther: leechh

metric - low is better
"""
import tensorflow as tf


def dice():
    def metric(y_true, y_pred):
        smooth = 1e-8
        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return 1 - ((2 * intersection) + smooth) / (union + smooth)
    return metric


def mean_dice():
    def metric(y_true, y_pred):
        smooth = 1e-8
        y_pred = tf.nn.sigmoid(y_pred)
        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0, 1, 2])
        union = tf.reduce_sum(y_true, axis=[0, 1, 2]) + tf.reduce_sum(y_pred, axis=[0, 1, 2])
        return tf.reduce_mean(1 - ((2 * intersection) + smooth) / (union + smooth))
    return metric


def softmax_ce():
    def metric(y_true, y_pred):
        return tf.losses.softmax_cross_entropy(y_true, y_pred)
    return metric


def sigmoid_ce():
    def metric(y_true, y_pred):
        return tf.losses.sigmoid_cross_entropy(y_true, y_pred)
    return metric


def focal(alpha=0.5, gamma=2):
    def metric(y_true, y_pred):
        # - https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/blob/master/focal_loss.py
        #y_pred = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, -1, 4])
        #y_true = tf.slice(y_true, [0, 0, 0, 0], [-1, -1, -1, 4])
        zeros = tf.zeros_like(y_pred, dtype=y_pred.dtype)

        # For poitive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        pos_p_sub = tf.where(y_true > zeros, y_true - y_pred, zeros)
        neg_p_sub = tf.where(y_true > zeros, zeros, y_pred)
        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(
            tf.clip_by_value(1.0 - y_pred, 1e-8, 1.0))
        return tf.reduce_sum(per_entry_cross_ent)
    return metric


def comboo_loss(params):
    def metric(y_true, y_pred):
        loss = tf.constant(0.)
        for metric_name, weight in params.items():
            if weight != 0:
                loss += (weight * metricfromname(metric_name, y_true, y_pred))
        return loss
    return metric


def metricfromname(metric_name, y_true, y_pred):
    # y_*: [batch, height, width, num_class]
    # dice
    if metric_name == 'dice':
        return dice()(y_true, y_pred)

    if metric_name == 'mean_dice':
        return mean_dice()(y_true, y_pred)

    if metric_name == 'softmax_ce':
        return softmax_ce()(y_true, y_pred)

    if metric_name == 'sigmoid_ce':
        return sigmoid_ce()(y_true, y_pred)

    if metric_name == 'focal':
        return focal()(y_true, y_pred)

    if metric_name == 'comboo_loss':
        return comboo_loss({'mean_dice': 1, 'sigmoid_ce': 1, 'focal': 1})(y_true, y_pred)
    else:
        assert False, 'metric function ISNOT exist'
