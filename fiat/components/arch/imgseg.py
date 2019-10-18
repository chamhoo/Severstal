"""
auther: leechh
"""
import tensorflow as tf
from ..arch.variable import Variable
from ..arch.layers import Layers


def deconv(node, c, filtersize=2, padding='SAME', folding=True, name='deconv'):
    """

    :param node: input feature map
    :param c: output channels
    :param padding:
    :param folding:
    :return:
    """
    if folding is True:
        input_c = int(2 * c)
        f = 2
    else:
        input_c = int(c)
        f = 1

    with tf.name_scope(name):
        wu = Variable.weight_variable(
            shape=[filtersize, filtersize, c, input_c],  # [filter, filter, output, input]
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(filtersize, c), 'mean': 0.},
            name='wu')
        bu = Variable.bias_variable(value=0, shape=[c], name='bu')
        node = Layers.deconv2d(node, wu, bu, strides=2, f=f, padding=padding, name='upconv')
    return tf.nn.relu(node)