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
            conv = tf.nn.batch_normalization(conv)
            return tf.nn.dropout(conv, rate=rate)

    @staticmethod
    def deconv2d(x, w, b, strids, padding='SAME', name='deconv2d'):
        with tf.name_scope(name):
            x_shape = tf.shape(x)
            output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
            deconv = tf.nn.conv2d_transpose(
                x, w,
                output_shape=output_shape,
                strides=[1, strids, strids, 1],
                padding=padding,
                name='upconv')
            return tf.nn.bias_add(deconv, b)

    @staticmethod
    def deconv2d_batch_norm(x, w, b, strids, padding='SAME', name='deconv2d_batch_norm'):
        deconv = Layers.deconv2d(x, w, b, strids, padding=padding, name=name)
        return tf.nn.batch_normalization(deconv)

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


class ResLayers(object):

    @staticmethod
    def identity_3layer(node, channels, rate=0.25):
        node_shortcut = node
        _, _, _, channel = tf.shape(node)
        channel1, channel2, channel3 = channels

        w1 = Variable.weight_variable(
            shape=[1, 1, channel, channel1],  # [filter, filter, input, output]
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(1, channel1), 'mean': 0.},
            name='w1')

        w2 = Variable.weight_variable(
            shape=[3, 3, channel1, channel2],  # [filter, filter, input, output]
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(3, channel2), 'mean': 0.},
            name='w2')

        w3 = Variable.weight_variable(
            shape=[1, 1, channel2, channel3],  # [filter, filter, input, output]
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(1, channel3), 'mean': 0.},
            name='w3')

        b1 = Variable.bias_variable(0, [channel1], name='b1')
        b2 = Variable.bias_variable(0, [channel2], name='b2')
        b3 = Variable.bias_variable(0, [channel3], name='b3')

        node = Layers.conv2d_batch_norm(node, w1, b1, padding='VALID', rate=rate)
        node = tf.nn.relu(node)
        node = Layers.conv2d_batch_norm(node, w2, b2, padding='SAME', rate=rate)
        node = tf.nn.relu(node)
        node = Layers.conv2d_batch_norm(node, w3, b3, padding='VALID', rate=rate)
        node = tf.math.add(node_shortcut, node)
        node = tf.nn.relu(node)
        return node

    @staticmethod
    def identity_2layer(node, channels, rate=0.25):
        node_shortcut = node
        _, _, _, channel = tf.shape(node)
        channel1, channel2 = channels

        w1 = Variable.weight_variable(
            shape=[3, 3, channel, channel1],  # [filter, filter, input, output]
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(3, channel1), 'mean': 0.},
            name='w1')

        w2 = Variable.weight_variable(
            shape=[3, 3, channel1, channel2],  # [filter, filter, input, output]
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(3, channel2), 'mean': 0.},
            name='w2')

        b1 = Variable.bias_variable(0, [channel1], name='b1')
        b2 = Variable.bias_variable(0, [channel2], name='b2')

        node = Layers.conv2d_batch_norm(node, w1, b1, padding='SAME', rate=rate)
        node = tf.nn.relu(node)
        node = Layers.conv2d_batch_norm(node, w2, b2, padding='SAME', rate=rate)
        node = tf.math.add(node_shortcut, node)
        node = tf.nn.relu(node)
        return node

    @staticmethod
    def convolutional_3layer(node, channels, rate=0.25, s=2):
        node_shortcut = node
        _, _, _, channel = tf.shape(node)
        channel1, channel2, channel3 = channels

        w_shortcut = Variable.weight_variable(
            shape=[1, 1, channel, channel3],  # [filter, filter, input, output]
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(1, channel3), 'mean': 0.},
            name='w_shortcut')

        w1 = Variable.weight_variable(
            shape=[1, 1, channel, channel1],  # [filter, filter, input, output]
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(1, channel1), 'mean': 0.},
            name='w1')

        w2 = Variable.weight_variable(
            shape=[3, 3, channel1, channel2],  # [filter, filter, input, output]
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(3, channel2), 'mean': 0.},
            name='w2')

        w3 = Variable.weight_variable(
            shape=[1, 1, channel2, channel3],  # [filter, filter, input, output]
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(1, channel3), 'mean': 0.},
            name='w3')

        b1 = Variable.bias_variable(0, [channel1], name='b1')
        b2 = Variable.bias_variable(0, [channel2], name='b2')
        b3 = Variable.bias_variable(0, [channel3], name='b3')
        b_shortcut = Variable.bias_variable(0, [channel3], name='b_shortcut')

        node = Layers.conv2d_batch_norm(node, w1, b1, padding='VALID', rate=rate, strides=s)
        node = tf.nn.relu(node)
        node = Layers.conv2d_batch_norm(node, w2, b2, padding='SAME', rate=rate)
        node = tf.nn.relu(node)
        node = Layers.conv2d_batch_norm(node, w3, b3, padding='VALID', rate=rate)
        node_shortcut = Layers.conv2d_batch_norm(
            x=node_shortcut,
            w=w_shortcut,
            b=b_shortcut,
            padding='VALID',
            rate=rate,
            strides=s)

        node = tf.math.add(node_shortcut, node)
        node = tf.nn.relu(node)
        return node

    @staticmethod
    def convolutional_2layer(node, channels, rate=0.25, s=2):
        node_shortcut = node
        _, _, _, channel = tf.shape(node)
        channel1, channel2 = channels

        w_shortcut = Variable.weight_variable(
            shape=[1, 1, channel, channel2],  # [filter, filter, input, output]
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(1, channel2), 'mean': 0.},
            name='w_shortcut')

        w1 = Variable.weight_variable(
            shape=[3, 3, channel, channel1],  # [filter, filter, input, output]
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(3, channel1), 'mean': 0.},
            name='w1')

        w2 = Variable.weight_variable(
            shape=[3, 3, channel1, channel2],  # [filter, filter, input, output]
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(3, channel2), 'mean': 0.},
            name='w2')

        b1 = Variable.bias_variable(0, [channel1], name='b1')
        b2 = Variable.bias_variable(0, [channel2], name='b2')
        b_shortcut = Variable.bias_variable(0, [channel2], name='b_shortcut')

        node = Layers.conv2d_batch_norm(node, w1, b1, padding='SAME', rate=rate, strides=s)
        node = tf.nn.relu(node)
        node = Layers.conv2d_batch_norm(node, w2, b2, padding='SAME', rate=rate)
        node_shortcut = Layers.conv2d_batch_norm(
            x=node_shortcut,
            w=w_shortcut,
            b=b_shortcut,
            padding='VALID',
            rate=rate,
            strides=s)

        node = tf.math.add(node_shortcut, node)
        node = tf.nn.relu(node)
        return node

    @staticmethod
    def layer2(node, level, num_layers=5, dropout_rate=0.25):
        # layer 2
        channel = 64
        with tf.name_scope('layer_2'):
            for i in range(3):
                node = ResLayers.identity_2layer(
                    node=node,
                    channels=[channel, channel],
                    rate=dropout_rate)

        # other layer
        for layer in range(3, num_layers + 1):
            channel *= 2
            with tf.name_scope(f'layer_{layer}'):

                # convolutional
                node = ResLayers.convolutional_2layer(
                    node=node,
                    channels=[channel, channel],
                    rate=dropout_rate)

                # identity
                for i in range(level[layer]):
                    node = ResLayers.identity_2layer(
                        node=node,
                        channels=[channel, channel],
                        rate=dropout_rate)
        return node

    @staticmethod
    def layer3(node, level, num_layers=5, dropout_rate=0.25):
        # layer 2
        channel = 64
        with tf.name_scope('layer_2'):
            for i in range(3):
                node = ResLayers.identity_3layer(
                    node=node,
                    channels=[channel, channel, channel],
                    rate=dropout_rate)

        # other layer
        # level = {3: 4 - 1, 4: 6 - 1, 5: 3 - 1}
        for layer in range(3, num_layers + 1):
            channel *= 2
            with tf.name_scope(f'layer_{layer}'):

                # convolutional
                node = ResLayers.convolutional_3layer(
                    node=node,
                    channels=[channel, channel, channel*4],
                    rate=dropout_rate)

                # identity
                for i in range(level[layer]):
                    node = ResLayers.identity_3layer(
                        node=node,
                        channels=[channel, channel, channel*4],
                        rate=dropout_rate)
        return node

    @staticmethod
    def layer(node, _type, num_layer=5, dropout_rate=0.5):

        structure = {
            18: {3: 1, 4: 1, 5: 1},
            34: {3: 3, 4: 5, 5: 2},
            50: {3: 3, 4: 5, 5: 2},
            101: {3: 3, 4: 22, 5: 2},
            152: {3: 7, 4: 35, 5: 2}
        }

        if _type in [18, 34]:
            node = ResLayers.layer2(node, structure[_type], num_layer, dropout_rate)
        else:
            node = ResLayers.layer3(node, structure[_type], num_layer, dropout_rate)
        return node