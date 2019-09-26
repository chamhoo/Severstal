import numpy as np
import tensorflow as tf
from tools.model_component import Layers, ModelComponent


class Model(Layers, ModelComponent):
    def unet(self, x, height, width, num_layers, feature_growth_rate, n_class, channels, padding, dropout_rate):
        node = x / 255
        down_node = {}
        self.model_name = 'U-Net'
        size_h, size_w = height, width

        # down path
        for layer in range(num_layers):
            with tf.name_scope(f'down_{layer}'):

                # features & stddev
                features = feature_growth_rate * (2 ** layer)
                stddev = np.sqrt(2 / (3**2 * features))

                # weight & bias
                if layer == 0:
                    w1 = self.weight_variable(
                        shape=[3, 3, channels, features],
                        type_name='truncated_normal',
                        weight_param={'stddev': stddev, 'mean': 0.},
                        name='w1')
                else:
                    w1 = self.weight_variable(
                        shape=[3, 3, features // 2, features],
                        type_name='truncated_normal',
                        weight_param={'stddev': stddev, 'mean': 0.},
                        name='w1')

                w2 = self.weight_variable(
                    shape=[3, 3, features, features],
                    type_name='truncated_normal',
                    weight_param={'stddev': stddev, 'mean': 0.},
                    name='w2')

                b1 = self.bias_variable(value=0, shape=[features], name='b1')
                b2 = self.bias_variable(value=0, shape=[features], name='b2')

                # down conv
                node = self.conv2d(node, w1, b1, padding=padding, name='conv1', rate=dropout_rate)
                node = tf.nn.relu(node)
                node = self.conv2d(node, w2, b2, padding=padding, name='conv2', rate=dropout_rate)
                node = tf.nn.relu(node)
                down_node[layer] = node

                # max pool & y size if VALID
                if padding == 'VALID':
                    size_h -= 2 * (3 - 1)   # conv_filter - 1
                    size_w -= 2 * (3 - 1)   # conv_filter - 1

                if layer < num_layers - 1:
                    node = self.maxpooling(node, n=2)
                    size_h /= 2
                    size_w /= 2

        for layer in range(num_layers - 2, -1, -1):
            with tf.name_scope(f'up_{layer}'):

                # features & stddev
                features = feature_growth_rate * (2 ** layer)
                stddev = np.sqrt(2 / (3 ** 2 * features))

                # weight & bias
                wu = self.weight_variable(
                        shape=[2, 2, features, 2 * features],   # [filter, filter, output, input]
                        type_name='truncated_normal',
                        weight_param={'stddev': stddev, 'mean': 0.},
                        name='wu')

                w1 = self.weight_variable(
                        shape=[3, 3, 2 * features, features],     # [filter, filter, input, output]
                        type_name='truncated_normal',
                        weight_param={'stddev': stddev, 'mean': 0.},
                        name='w1')

                w2 = self.weight_variable(
                        shape=[3, 3, features, features],
                        type_name='truncated_normal',
                        weight_param={'stddev': stddev, 'mean': 0.},
                        name='w2')

                bu = self.bias_variable(value=0, shape=[features], name='bu')
                b1 = self.bias_variable(value=0, shape=[features], name='b1')
                b2 = self.bias_variable(value=0, shape=[features], name='b2')

                # de conv
                node = self.deconv2d(node, wu, bu, strids=2, padding=padding, name='upconv')
                node = tf.nn.relu(node)
                node = self.concat(down_node[layer], node)

                node = self.conv2d(node, w1, b1, padding=padding, name='conv1', rate=dropout_rate)
                node = tf.nn.relu(node)
                node = self.conv2d(node, w2, b2, padding=padding, name='conv2', rate=dropout_rate)
                node = tf.nn.relu(node)

                # y size if VALID
                size_h *= 2
                size_w *= 2
                if padding == 'VALID':
                    size_h -= 2 * (3 - 1)
                    size_w -= 2 * (3 - 1)

        with tf.name_scope('output'):

            # weight & bias
            w = self.weight_variable(
                shape=[1, 1, feature_growth_rate, n_class],
                type_name='truncated_normal',
                weight_param={'stddev': stddev, 'mean': 0.},
                name='output_w')

            b = self.bias_variable(value=0, shape=[n_class], name='output_b')

            # output
            node = self.conv2d(node, w, b, padding=padding, name='output_conv', rate=0)
            # node = tf.nn.softmax(node)
            # node = tf.nn.relu(node)
        return node, int(size_h), int(size_w)