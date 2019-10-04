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
            conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding=padding)
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

    def dice(self, y_true, y_pred):
        smooth = 1e-8
        y_pred = tf.nn.softmax(y_pred)
        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return ((2 * intersection) + smooth) / (union + smooth)

    def dice_severstal(self, y_true, y_pred):
        smooth = 1e-8
        y_pred = tf.nn.softmax(y_pred)
        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0, 1, 2])
        union = tf.reduce_sum(y_true, axis=[0, 1, 2]) \
                + tf.reduce_sum(y_pred, axis=[0, 1, 2])
        return - tf.reduce_mean(
            tf.slice(((2 * intersection) + smooth) / (union + smooth), [0], [4]))

    def bce_severstal(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred)
        y_pred = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, -1, 4])
        y_true = tf.slice(y_true, [0, 0, 0, 0], [-1, -1, -1, 4])

        yadd = tf.multiply(y_true, tf.math.log1p(y_pred))
        ysub = tf.multiply((1 - y_true), tf.math.log1p((1 - y_pred)))
        return tf.reduce_mean(tf.reduce_sum((-(yadd + ysub)), axis=[0, 1, 2]))

    def focal_severstal(self, y_true, y_pred, alpha=0.5, gamma=2):
        # - https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/blob/master/focal_loss.py
        y_pred = tf.nn.softmax(y_pred)
        y_pred = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, -1, 4])
        y_true = tf.slice(y_true, [0, 0, 0, 0], [-1, -1, -1, 4])
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
        return tf.reduce_mean(tf.reduce_sum(per_entry_cross_ent, axis=[0, 1, 2]))

    def comboo_loss(self, y_true, y_pred, params):
        loss = tf.constant(0.)
        for metric_name, weight in params.items():
            if weight != 0:
                loss += (weight * self.metric_func(metric_name, y_true, y_pred))
        return loss

    def metric_func(self, metric_name, y_true, y_pred, params=None):
        # y_*: [batch, height, width, num_class]
        # dice
        if metric_name == 'dice':
            return self.dice(y_true, y_pred)
        
        if metric_name == 'bce':
            return self.bce_severstal(y_true, y_pred)

        if metric_name == 'focal':
            return self.metric_func(y_true, y_pred)

        if metric_name == 'comboo_loss':
            return self.comboo_loss(y_true, y_pred, params)
        else:
            assert False, 'metric function ISNOT exist'