"""
auther: leechh
"""
import tensorflow as tf
from math import sqrt
from layer import Layers


def unet(x, num_layers=5, feature_growth_rate=64, n_class=1, channels=3, padding='SAME', dropout_rate=0.25):
    node = x / 255
    down_node = {}
    layers = Layers()

    # down path
    for layer in range(num_layers):
        with tf.name_scope(f'down_{layer}'):

            # features & stddev
            features = feature_growth_rate * (2 ** layer)
            stddev = sqrt(2 / (3**2 * features))

            # weight & bias
            if layer == 0:
                w1 = layers.weight_variable(
                    shape=[3, 3, channels, features],
                    type_name='truncated_normal',
                    weight_param={'stddev': stddev, 'mean': 0.},
                    name='w1')
            else:
                w1 = layers.weight_variable(
                    shape=[3, 3, features // 2, features],
                    type_name='truncated_normal',
                    weight_param={'stddev': stddev, 'mean': 0.},
                    name='w1')

            w2 = layers.weight_variable(
                shape=[3, 3, features, features],
                type_name='truncated_normal',
                weight_param={'stddev': stddev, 'mean': 0.},
                name='w2')

            b1 = layers.bias_variable(value=0, shape=[features], name='b1')
            b2 = layers.bias_variable(value=0, shape=[features], name='b2')

            # down conv
            node = layers.conv2d(node, w1, b1, padding=padding, name='conv1', rate=dropout_rate)
            node = tf.nn.relu(node)
            node = layers.conv2d(node, w2, b2, padding=padding, name='conv2', rate=dropout_rate)
            node = tf.nn.relu(node)
            down_node[layer] = node

            # max pool
            if layer < num_layers - 1:
                node = layers.maxpooling(node, n=2)

    for layer in range(num_layers - 2, -1, -1):
        with tf.name_scope(f'up_{layer}'):

            # features & stddev
            features = feature_growth_rate * (2 ** layer)
            stddev = sqrt(2 / (3 ** 2 * features))

            # weight & bias
            wu = layers.weight_variable(
                    shape=[2, 2, features, 2 * features],   # [filter, filter, output, input]
                    type_name='truncated_normal',
                    weight_param={'stddev': stddev, 'mean': 0.},
                    name='wu')

            w1 = layers.weight_variable(
                    shape=[3, 3, 2 * features, features],     # [filter, filter, input, output]
                    type_name='truncated_normal',
                    weight_param={'stddev': stddev, 'mean': 0.},
                    name='w1')

            w2 = layers.weight_variable(
                    shape=[3, 3, features, features],
                    type_name='truncated_normal',
                    weight_param={'stddev': stddev, 'mean': 0.},
                    name='w2')

            bu = layers.bias_variable(value=0, shape=[features], name='bu')
            b1 = layers.bias_variable(value=0, shape=[features], name='b1')
            b2 = layers.bias_variable(value=0, shape=[features], name='b2')

            # de conv
            node = layers.deconv2d(node, wu, bu, strids=2, padding=padding, name='upconv')
            node = tf.nn.relu(node)
            node = layers.concat(down_node[layer], node)

            node = layers.conv2d(node, w1, b1, padding=padding, name='conv1', rate=dropout_rate)
            node = tf.nn.relu(node)
            node = layers.conv2d(node, w2, b2, padding=padding, name='conv2', rate=dropout_rate)
            node = tf.nn.relu(node)

    with tf.name_scope('output'):

        # weight & bias
        w = layers.weight_variable(
            shape=[1, 1, feature_growth_rate, n_class],
            type_name='truncated_normal',
            weight_param={'stddev': stddev, 'mean': 0.},
            name='output_w')

        b = layers.bias_variable(value=0, shape=[n_class], name='output_b')

        # output
        node = layers.conv2d(node, w, b, padding=padding, name='output_conv', rate=0)
    return node
