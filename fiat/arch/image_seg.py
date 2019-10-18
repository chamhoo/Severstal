"""
auther: leechh
"""
import tensorflow as tf
from math import sqrt
from ..components.arch.layers import Layers
from ..components.arch.variable import Variable
from ..components.arch.res import structure, ResNet, ResNext, SeResNet, SeResNext
from ..components.arch.imgseg import deconv


def unet(num_layers=5, feature_growth_rate=64, n_class=1, channels=3,
         padding='SAME', dropout_rate=0.25, active='sigmoid'):

    def model(x):
        node = x / 255
        down_node = {}
        # down path
        for layer in range(1, num_layers + 1):
            with tf.name_scope(f'down_{layer}'):

                # features & stddev
                features = feature_growth_rate * (2 ** (layer - 1))
                stddev = sqrt(2 / (3**2 * features))

                # weight & bias
                if layer == 1:
                    w1 = Variable.weight_variable(
                        shape=[3, 3, channels, features],
                        type_name='truncated_normal',
                        weight_param={'stddev': stddev, 'mean': 0.},
                        name=f'w1_{layer}')
                else:
                    w1 = Variable.weight_variable(
                        shape=[3, 3, features // 2, features],
                        type_name='truncated_normal',
                        weight_param={'stddev': stddev, 'mean': 0.},
                        name=f'w1_{layer}')

                w2 = Variable.weight_variable(
                    shape=[3, 3, features, features],
                    type_name='truncated_normal',
                    weight_param={'stddev': stddev, 'mean': 0.},
                    name=f'w2_{layer}')

                b1 = Variable.bias_variable(value=0, shape=[features], name=f'b1_{layer}')
                b2 = Variable.bias_variable(value=0, shape=[features], name=f'b2_{layer}')

                # down conv
                node = Layers.conv2d(node, w1, b1, padding=padding, name=f'conv1_{layer}', rate=dropout_rate)
                node = tf.nn.relu(node)
                node = Layers.conv2d(node, w2, b2, padding=padding, name=f'conv2_{layer}', rate=dropout_rate)
                node = tf.nn.relu(node)
                down_node[layer] = node

                # max pool
                if layer < num_layers:
                    node = Layers.maxpooling(node, n=2)

        for layer in range(num_layers - 1, 0, -1):
            with tf.name_scope(f'up_{layer}'):

                # features & stddev
                features = feature_growth_rate * (2 ** (layer - 1))
                stddev = sqrt(2 / (3 ** 2 * features))

                # weight & bias
                wu = Variable.weight_variable(
                        shape=[2, 2, features, 2 * features],   # [filter, filter, output, input]
                        type_name='truncated_normal',
                        weight_param={'stddev': stddev, 'mean': 0.},
                        name='wu')

                w1 = Variable.weight_variable(
                        shape=[3, 3, 2 * features, features],     # [filter, filter, input, output]
                        type_name='truncated_normal',
                        weight_param={'stddev': stddev, 'mean': 0.},
                        name='w1')

                w2 = Variable.weight_variable(
                        shape=[3, 3, features, features],
                        type_name='truncated_normal',
                        weight_param={'stddev': stddev, 'mean': 0.},
                        name='w2')

                bu = Variable.bias_variable(value=0, shape=[features], name='bu')
                b1 = Variable.bias_variable(value=0, shape=[features], name='b1')
                b2 = Variable.bias_variable(value=0, shape=[features], name='b2')

                # de conv
                node = Layers.deconv2d(node, wu, bu, strides=2, padding=padding, name='upconv')
                node = tf.nn.relu(node)
                node = Layers.concat(down_node[layer], node)

                node = Layers.conv2d(node, w1, b1, padding=padding, name='conv1', rate=dropout_rate)
                node = tf.nn.relu(node)
                node = Layers.conv2d(node, w2, b2, padding=padding, name='conv2', rate=dropout_rate)
                node = tf.nn.relu(node)

        with tf.name_scope('output'):

            # weight & bias
            w = Variable.weight_variable(
                shape=[1, 1, feature_growth_rate, n_class],
                type_name='truncated_normal',
                weight_param={'stddev': stddev, 'mean': 0.},
                name='output_w')

            b = Variable.bias_variable(value=0, shape=[n_class], name='output_b')

            # output
            node = Layers.conv2d(node, w, b, padding=padding, name='output_conv', rate=0)
            if active == 'sigmoid':
                node = tf.nn.sigmoid(node)
            elif active == 'softmax':
                node = tf.nn.softmax(node)

        return node
    return model


def resunet_arch(reslayer, numlayers='50', numstages=5, channels=64,
                 n_class=1, input_channel=3, padding='SAME', rate=0.25, active='sigmoid'):

    if type(reslayer) == str:
        if reslayer in ['resnet', 'Resnet', 'ResNet']:
            reslayer = ResNet
        elif reslayer in ['resnext', 'Resnext', 'ResNext']:
            reslayer = ResNext
        elif reslayer in ['seresnet', 'SerRsnet', 'SeResNet']:
            reslayer = SeResNet
        elif reslayer in ['seresnext', 'SeResnext', 'SEResNext']:
            reslayer = SeResNext
        else:
            assert False, 'not exist'

    def model(x):

        node = x / 255
        down_node = {}

        if numlayers in ['18', '34']:
            num_blocklayer = 2
            growth = 1
        else:
            num_blocklayer = 3
            growth = 4

        # down path
        # downlayer 1
        c = channels * 1
        with tf.name_scope('down_1'):
            node = reslayer(node, input_channel, c, num_blocklayer, s=2, rate=rate).firstlayer()
            down_node[1] = node

        # downlayer 2
        c = c * 1
        with tf.name_scope('down_2'):
            node = Layers.maxpooling(node, n=3, s=2)
            node = reslayer(node, c, c*growth, num_blocklayer, s=1, rate=rate).block(name=f'block_first')
            for i in range(structure[numlayers][2] - 1):
                node = reslayer(node, c*growth, c*growth, num_blocklayer, s=1, rate=rate).block(name=f'block{i}')
            down_node[2] = node

        # down other layer
        for stage in range(3, numstages + 1):
            c = c * 2
            with tf.name_scope(f'down_{stage}'):
                node = reslayer(node, int(c/2)*growth, c*growth, num_blocklayer, s=2, rate=rate).block(name='block_down')
                for i in range(structure[numlayers][stage] - 1):
                    node = reslayer(node, c*growth, c*growth, num_blocklayer, s=1, rate=rate).block(name=f'block{i}')
                # copy
                if stage != numstages:
                    down_node[stage] = node

        # up other path
        for stage in range(numstages-1, 1, -1):
            with tf.name_scope(f'up_{stage}'):
                c = int(c / 2)
                # concat & deconv & f/2
                node = deconv(node, c*growth)
                node = Layers.concat(down_node[stage], node)
                node = reslayer(node, 2*c*growth, c*growth, num_blocklayer, s=1, rate=rate).block(name='upstage')

                for i in list(range(structure[numlayers][stage] - 1))[::-1]:
                    if (stage == 2) & (i == 0):
                        node = reslayer(node, c*growth, c, 2, s=1, rate=rate).block(name=f'block{i}')
                    else:
                        node = reslayer(node, c*growth, c*growth, num_blocklayer, s=1, rate=rate).block(name=f'block{i}')

        # up 1
        with tf.name_scope('up_1'):
            # upsample
            node = deconv(node, c, folding=False, name='upsample')
            # deconv
            c = int(c / 2)
            node = deconv(node, c, filtersize=7, name='deconv7')
            # conv
            with tf.name_scope('conv'):
                w = Variable.weight_variable(
                    shape=[3, 3, c, c],
                    type_name='truncated_normal',
                    weight_param={'stddev': Variable.cal_std(3, c), 'mean': 0.},
                    name='output_w')
                b = Variable.bias_variable(value=0, shape=[c], name='output_b')
                node = Layers.conv2d(node, w, b, padding=padding, name='output_conv', rate=0)
                node = tf.nn.relu(node)

        # output
        with tf.name_scope('output'):
            # output
            w_output = Variable.weight_variable(
                shape=[1, 1, c, n_class],
                type_name='truncated_normal',
                weight_param={'stddev': Variable.cal_std(1, n_class), 'mean': 0.},
                name='output_w')
            b_output = Variable.bias_variable(value=0, shape=[n_class], name='output_b')
            node = Layers.conv2d(node, w_output, b_output, padding=padding, name='output_conv', rate=0)

            if active == 'sigmoid':
                node = tf.nn.sigmoid(node)
            elif active == 'softmax':
                node = tf.nn.softmax(node)
        return node
    return model
