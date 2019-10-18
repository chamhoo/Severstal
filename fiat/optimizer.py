"""
auther: leechh
"""
import tensorflow as tf


def adam(learning_rate, loss):
    return tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ).minimize(loss)


def adagrad(learning_rate, loss):
    return tf.train.AdagradOptimizer(
        learning_rate=learning_rate,
        initial_accumulator_value=1e-8
    ).minimize(loss)


def gd(learning_rate, loss):
    return tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate,
    ).minimize(loss)


def momentun(learning_rate, loss):
    return tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
         momentum=0.99
    ).minimize(loss)


def optfromname(name, **kwargs):
    if name == 'adam':
        return adam(kwargs['learning_rate'], kwargs['loss'])
    elif name == 'adagrad':
        return adagrad(kwargs['learning_rate'], kwargs['loss'])
    elif name == 'gd':
        return gd(kwargs['learning_rate'], kwargs['loss'])
    elif name == 'momentun':
        return momentun(kwargs['learning_rate'], kwargs['loss'])
    else:
        assert False, 'optimizer name ISNOT exist'