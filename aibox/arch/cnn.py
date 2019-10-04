from math import sqrt
import tensorflow as tf
from aibox.arch.layer import Variable, Layers, ResLayers


def resnet(x, _type=34, num_layers=5, n_class=1, dropout_rate=0.25):
    """

    :param x:
    :param _type:
    :param num_layers:
    :param n_class:
    :param dropout_rate:
    :return:
    """
    # input
    node = x
    batch_size, height, widht, channel = tf.shape(x)

    # layer 1
    with tf.name_scope('layer_1'):
        stddev = sqrt(2 / (7**2 * 64))

        w1 = Variable.weight_variable(
            shape=(7, 7, channel, 64),
            type_name='truncated_normal',
            weight_param={'stddev': stddev, 'mean': 0.},
            name='w1')

        b1 = Variable.bias_variable(value=0, shape=[64], name='b1')

        node = Layers.conv2d_batch_norm(node, w1, b1, padding='SAME', rate=dropout_rate)
        node = tf.nn.relu(node)
        node = Layers.maxpooling(node, n=3, s=2)

    # other layer
    node = ResLayers.layer(node, _type, num_layer=num_layers, dropout_rate=dropout_rate)

    # AveragePooling & FC
    node = tf.math.reduce_mean(node, axis=[1, 2], keepdims=True)
    node = tf.reshape(node, shape=[batch_size, -1])

    w = Variable.weight_variable(
        shape=[batch_size, n_class],
        type_name='truncated_normal',
        weight_param={'stddev': stddev, 'mean': 0.},
        name='w_fc')

    b = Variable.bias_variable(0, shape=[1], name='b_fc')

    node = tf.add(tf.matmul(node, w), b)  # [batch_size, n_class]
    return node


