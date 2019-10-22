"""
auther: leechh

loss - low is better
"""
import tensorflow as tf

def tversky(y_true, y_pred, smooth=1e-6):
    y_true_pos = tf.keras.layers.Flatten()(y_true)
    y_pred_pos = tf.keras.layers.Flatten()(y_pred)
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.75
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)


def tversky_loss():
    def loss(y_true, y_pred):
        return 1 - tversky(y_true,y_pred)
    return loss

def focal_tversky_loss(gamma = 0.75):
    def loss(y_true,y_pred):
        pt_1 = tversky(y_true, y_pred)
        return tf.keras.backend.pow((1-pt_1), gamma)
    return loss

def dice():
    def loss(y_true, y_pred):
        smooth = 1e-8
        y_true = tf.keras.backend.flatten(y_true)
        y_pred = tf.keras.backend.flatten(y_pred)
        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return 1 - ((2 * intersection) + smooth) / (union + smooth)
    return loss


def mean_dice(axis=[1, 2]):
    def loss(y_true, y_pred):
        smooth = 1e-8
        y_pred = tf.nn.sigmoid(y_pred)
        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=axis)
        union = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis)
        return 1 - tf.reduce_mean(((2 * intersection) + smooth) / (union + smooth))
    return loss

def softmax_ce():
    def loss(y_true, y_pred):
        return tf.losses.softmax_cross_entropy(y_true, y_pred)
    return loss


def sigmoid_ce():
    def loss(y_true, y_pred):
        return tf.losses.sigmoid_cross_entropy(y_true, y_pred)
    return loss

def bce(active=None):
    return tf.keras.losses.BinaryCrossentropy()


def focal(alpha=0.5, gamma=2):
    def loss(y_true, y_pred):
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
    return loss


def comboo_loss(params):
    def loss(y_true, y_pred):
        loss = tf.constant(0.)
        for metric_name, weight in params.items():
            if weight != 0:
                loss += (weight * metricfromname(metric_name, y_true, y_pred))
        return loss
    return loss


def lossfromname(loss_name, y_true, y_pred):
    # y_*: [batch, height, width, num_class]
    # dice
    if loss_name == 'dice':
        return dice()(y_true, y_pred)

    if loss_name == 'mean_dice':
        return mean_dice()(y_true, y_pred)
    
    if loss_name == 'bce':
        return bce()(y_true, y_pred)

    if loss_name == 'softmax_ce':
        return softmax_ce()(y_true, y_pred)

    if loss_name == 'sigmoid_ce':
        return sigmoid_ce()(y_true, y_pred)

    if loss_name == 'focal':
        return focal()(y_true, y_pred)

    if loss_name == 'comboo_loss':
        return comboo_loss({'mean_dice': 1, 'sigmoid_ce': 1, 'focal': 1})(y_true, y_pred)
    else:
        assert False, 'loss function ISNOT exist'