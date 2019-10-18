"""
auther: leechh
"""
import tensorflow as tf
from math import sqrt
from fiat.components.arch.variable import Variable
from fiat.components.arch.layers import Layers


structure = {
    '18': {2: 2, 3: 2, 4: 2, 5: 2},
    '34': {2: 3, 3: 4, 4: 6, 5: 3},
    '50': {2: 3, 3: 4, 4: 6, 5: 3},
    '101': {2: 3, 3: 4, 4: 23, 5: 3},
    '152': {2: 3, 3: 8, 4: 36, 5: 3},
}


class ResClassifier(object):

    def __init__(self, x, net, input_c, output_c, numlayer='18', nstage=5, nclass=1, dropout_rate=0.25):
        """

        :param x:
        :param net:
        :param numlayer: '18', '34', '50', '101', '152'
        :param nstage:
        :param nclass:
        :param dropout_rate:
        """
        self.netparam = {'rate': dropout_rate}
        self.node = x
        self.net = net
        self.numlayer = numlayer
        self.nstage = nstage
        self.nclass = nclass
        self.output_c = output_c
        self.input_c = input_c
        self.shape = tf.shape(x)
        self.structure = structure

        if numlayer in ['18', '34']:
            self.netparam['num_blocklayer'] = 2
        else:
            self.netparam['num_blocklayer'] = 3

    def reslayer1(self):
        # layer 1
        with tf.name_scope('layer_1'):
            self.node = self.net(self.node, self.output_c, s=2, **self.netparam).firstlayer()
            self.node = Layers.maxpooling(self.node, n=3, s=2)

    def resmainlayer(self):
        # layer 2
        with tf.name_scope('layer_2'):
            for i in range(self.structure[self.numlayer][2]):
                self.node = self.net(self.node, self.output_c, s=1, **self.netparam).block()

        # other layer
        for layer in range(3, self.nstage + 1):
            self.output_c *= 2
            with tf.name_scope(f'layer_{layer}'):
                # convolutional
                self.node = self.net(self.node, self.output_c, s=2).block()
                # identity
                for i in range(self.structure[self.numlayer][layer] - 1):
                    self.node = self.net(self.node, self.output_c, s=1).block()

    def resoutput(self):
        # AveragePooling
        self.node = tf.math.reduce_mean(self.node, axis=[1, 2], keepdims=False)
        shape = tf.shape(self.node)
        # FC
        w_fc = Variable.weight_variable(
            shape=[shape[3], self.nclass],
            type_name='truncated_normal',
            weight_param={'stddev': sqrt(2 / shape[3] * self.nclass), 'mean': 0.},
            name='w_fc')

        b_fc = Variable.bias_variable(0, shape=[1], name='b_fc')
        return tf.add(tf.matmul(self.node, w_fc), b_fc)  # [batch_size, n_class]


class ResLayers(object):

    def __init__(self, node, input_c, output_c, num_blocklayer, s=2, rate=0.25):
        self.node = node
        self.node_shortcut = self.node
        self.output_c = int(output_c)
        self.input_c = input_c
        self.num_blocklayer = int(num_blocklayer)
        self.s = int(s)
        self.rate = rate
        self.shape = tf.shape(node)

    def firstlayer(self):
        w1 = Variable.weight_variable(
            shape=(7, 7, self.input_c, self.output_c),
            type_name='truncated_normal',
            weight_param={'stddev': Variable.cal_std(7, self.output_c), 'mean': 0.},
            name='w1')

        b1 = Variable.bias_variable(value=0, shape=[self.output_c], name='b1')

        self.node = Layers.conv2d_batch_norm(self.node, w1, b1, padding='SAME', strides=self.s, rate=self.rate)
        return tf.nn.relu(self.node)

    def add(self):
        self.node = tf.math.add(self.node_shortcut, self.node)
        self.node = tf.nn.relu(self.node)
        return tf.layers.batch_normalization(self.node)

    def shortcut(self):
        if (self.input_c != self.output_c) or (self.s != 1):
            w_shortcut = Variable.weight_variable(
                shape=[1, 1, self.input_c, self.output_c],  # [filter, filter, input, output]
                type_name='truncated_normal',
                weight_param={'stddev': Variable.cal_std(1, self.output_c), 'mean': 0.},
                name='w_shortcut')
            b_shortcut = Variable.bias_variable(0, [self.output_c], name='b_shortcut')
            self.node_shortcut = Layers.conv2d_batch_norm(
                x=self.node_shortcut,
                w=w_shortcut,
                b=b_shortcut,
                padding='VALID',
                rate=self.rate,
                strides=self.s)
            self.node_shortcut = tf.nn.relu(self.node_shortcut)


class ResNetBlock(ResLayers):

    def resnet_block(self):
        if self.num_blocklayer == 2:

            w1 = Variable.weight_variable(
                shape=[3, 3, self.input_c, self.output_c],  # [filter, filter, input, output]
                type_name='truncated_normal',
                weight_param={'stddev': Variable.cal_std(3, self.output_c), 'mean': 0.},
                name='w1')
            b1 = Variable.bias_variable(0, [self.output_c], name='b1')
            self.node = Layers.conv2d_batch_norm(
                x=self.node,
                w=w1,
                b=b1,
                padding='SAME',
                rate=self.rate,
                strides=self.s)

            self.node = tf.nn.relu(self.node)

            w2 = Variable.weight_variable(
                shape=[3, 3, self.output_c, self.output_c],  # [f, f, input_c/4, output_c/4]
                type_name='truncated_normal',
                weight_param={'stddev': Variable.cal_std(3, self.output_c), 'mean': 0.},
                name='w2')
            b2 = Variable.bias_variable(0, [self.output_c], name='b2')
            self.node = Layers.conv2d_batch_norm(
                x=self.node,
                w=w2,
                b=b2,
                padding='SAME',
                rate=self.rate)
            self.node = tf.nn.relu(self.node)

        else:
            w1 = Variable.weight_variable(
                shape=[1, 1, self.input_c, int(self.output_c / 4)],  # [filter, filter, input, output]
                type_name='truncated_normal',
                weight_param={'stddev': Variable.cal_std(1, self.output_c / 4), 'mean': 0.},
                name='w1')
            b1 = Variable.bias_variable(0, [int(self.output_c / 4)], name='b1')
            self.node = Layers.conv2d_batch_norm(
                x=self.node,
                w=w1,
                b=b1,
                padding='VALID',
                rate=self.rate,
                strides=self.s)
            self.node = tf.nn.relu(self.node)

            w2 = Variable.weight_variable(
                shape=[3, 3, int(self.output_c/4), int(self.output_c/4)],  # [filter, filter, input, output]
                type_name='truncated_normal',
                weight_param={'stddev': Variable.cal_std(3, self.output_c/4), 'mean': 0.},
                name='w2')
            b2 = Variable.bias_variable(0, [int(self.output_c / 4)], name='b2')
            self.node = Layers.conv2d_batch_norm(
                x=self.node,
                w=w2,
                b=b2,
                padding='SAME',
                rate=self.rate)
            self.node = tf.nn.relu(self.node)

            w3 = Variable.weight_variable(
                shape=[1, 1, int(self.output_c / 4), self.output_c],  # [filter, filter, input, output]
                type_name='truncated_normal',
                weight_param={'stddev': Variable.cal_std(1, self.output_c), 'mean': 0.},
                name='w3')
            b3 = Variable.bias_variable(0, [self.output_c], name='b3')
            self.node = Layers.conv2d_batch_norm(
                x=self.node,
                w=w3,
                b=b3,
                padding='VALID',
                rate=self.rate)
            self.node = tf.nn.relu(self.node)


class ResNet(ResNetBlock):

    def block(self, name='resnet_block'):
        with tf.name_scope(name):
            self.resnet_block()
            self.shortcut()
            self.add()
        return self.node


class ResNextBlock(ResLayers):

    def resnext_block(self, c=4):   # c=32
        node_lst = []
        if self.num_blocklayer == 2:
            output_c = int(self.output_c / c)
            for i in range(c):
                with tf.name_scope(f'path{i}'):
                    w1 = Variable.weight_variable(
                        shape=[3, 3, self.input_c, output_c],  # [filter, filter, input, output]
                        type_name='truncated_normal',
                        weight_param={'stddev': Variable.cal_std(3, output_c), 'mean': 0.},
                        name='w1')
                    b1 = Variable.bias_variable(0, [output_c], name='b1')
                    node = Layers.conv2d_batch_norm(
                        x=self.node,
                        w=w1,
                        b=b1,
                        padding='SAME',
                        rate=self.rate,
                        strides=self.s)

                    node = tf.nn.relu(node)

                    w2 = Variable.weight_variable(
                        shape=[3, 3, output_c, output_c],  # [filter, filter, input, output]
                        type_name='truncated_normal',
                        weight_param={'stddev': Variable.cal_std(3, output_c), 'mean': 0.},
                        name='w2')
                    b2 = Variable.bias_variable(0, [output_c], name='b2')

                    node = Layers.conv2d_batch_norm(
                        x=node,
                        w=w2,
                        b=b2,
                        padding='SAME',
                        rate=self.rate)
                    node = tf.nn.relu(node)
                    node_lst.append(node)

            self.node = tf.concat(node_lst, axis=3)

        else:
            output_c = int(self.output_c / c)
            for i in range(c):
                with tf.name_scope(f'path{i}'):
                    w1 = Variable.weight_variable(
                        shape=[1, 1, self.input_c, int(output_c / 2)],  # [filter, filter, input, output]
                        type_name='truncated_normal',
                        weight_param={'stddev': Variable.cal_std(1, output_c / 2), 'mean': 0.},
                        name='w1')

                    w2 = Variable.weight_variable(
                        shape=[3, 3, int(output_c / 2), int(output_c / 2)],  # [filter, filter, input, output]
                        type_name='truncated_normal',
                        weight_param={'stddev': Variable.cal_std(3, output_c), 'mean': 0.},
                        name='w2')

                    b1 = Variable.bias_variable(0, [int(output_c / 2)], name='b1')
                    b2 = Variable.bias_variable(0, [int(output_c / 2)], name='b2')

                    node = Layers.conv2d_batch_norm(
                        x=self.node,
                        w=w1,
                        b=b1,
                        padding='VALID',
                        rate=self.rate,
                        strides=self.s)
                    node = tf.nn.relu(node)
                    node = Layers.conv2d_batch_norm(
                        x=node,
                        w=w2,
                        b=b2,
                        padding='SAME',
                        rate=self.rate)
                    node = tf.nn.relu(node)
                    node_lst.append(node)

            # concat
            node = tf.concat(node_lst, axis=3)

            w3 = Variable.weight_variable(
                shape=[1, 1, int(output_c / 2), self.output_c],  # [filter, filter, input, output]
                type_name='truncated_normal',
                weight_param={'stddev': Variable.cal_std(1, self.output_c), 'mean': 0.},
                name='w3')

            b3 = Variable.bias_variable(0, [self.output_c], name='b3')

            node = Layers.conv2d_batch_norm(
                x=node,
                w=w3,
                b=b3,
                padding='VALID',
                rate=self.rate)
            self.node = tf.nn.relu(node)


class ResNext(ResNextBlock):

    def block(self, name='resnext_block'):
        with tf.name_scope(name):
            self.resnext_block()
            self.shortcut()
            self.add()
        return self.node


def senet(node, output_c, r=16):  # r=16
    with tf.name_scope('senet'):
        # Squeeze
        node = tf.math.reduce_mean(node, axis=[1, 2], keepdims=False)  # node [batch, 1, 1, c]

        # Excitation
        w1 = Variable.weight_variable(
            shape=[output_c, int(output_c / r)],
            type_name='truncated_normal',
            weight_param={'stddev': sqrt(2 / output_c * (output_c / r)), 'mean': 0.},
            name='w1')

        b1 = Variable.bias_variable(0, shape=[1], name='b1')

        w2 = Variable.weight_variable(
            shape=[int(output_c / r), output_c],
            type_name='truncated_normal',
            weight_param={'stddev': sqrt(2 / output_c * (output_c / r)), 'mean': 0.},
            name='w2')

        b2 = Variable.bias_variable(0, shape=[1], name='b2')

        node = tf.add(tf.matmul(node, w1), b1)
        node = tf.nn.relu(node)
        node = tf.add(tf.matmul(node, w2), b2)
    return tf.math.sigmoid(node)   # [batch_size, c]


class SeResNet(ResNetBlock):

    def block(self, name='seresnet_block'):
        with tf.name_scope(name):
            # residual
            self.resnet_block()
            residual = self.node   # [batch_size, h, w, output_c]
            # senet
            self.node = senet(self.node, output_c=self.output_c)  # [batch, 1, 1, output_c]
            # Reweight
            self.node = tf.multiply(self.node, residual)
            # add
            self.shortcut()
            self.add()
        return self.node


class SeResNext(ResNextBlock):

    def block(self, name='seresnext_block'):
        with tf.name_scope(name):
            # residual
            self.resnext_block()
            residual = self.node   # [batch_size, h, w, output_c]
            # senet
            self.node = senet(self.node, output_c=self.output_c)
            # Reweight
            self.node = tf.multiply(self.node, residual)
            # add
            self.shortcut()
            self.add()
        return self.node
