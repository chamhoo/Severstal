"""
auther: leechh
"""
import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from fiat.chunk import chunk
from fiat.logging import logging
from fiat.os import mkdir
from fiat.components.train.train import empty_recorder, checkpoint, cal_mean
from fiat.metric import metricfromname, dice
from fiat.optimizer import optfromname
from fiat.data import TFR


def train(arch, datafunc, loss, metric, optimizer, rate, epoch=1, batch_size=32, early_stopping=5,
          verbose=2, retrain=False, reshape=None, reshape_method=None, ckpt_path=None, distrainable=False):
    """

    :param arch:
    :param datafunc: {'train': tensorflow dataset, 'valid': tensorflow dataset} or tensorflow dataset
    :param loss:
    :param metric:
    :param optimizer:
    :param rate:
    :param epoch:
    :param batch_size:
    :param early_stopping:
    :param verbose:
    :param retrain:False or dict, dict key is 'start_epoch', 'ckpt', 'train_loss', 'valid_metrics'
    :param reshape:
    :param reshape_method:
    :param ckpt_path:
    :return:
    """
    def back_operation():
        with tf.name_scope('backpropagation'):
            if type(loss) is str:
                loss_back = metricfromname(loss, y_true, y_pred)
            else:
                loss_back = loss(y_true, y_pred)
            # opt
            if type(optimizer) is str:
                opt_back = optfromname(optimizer, learning_rate=rate, loss=loss_back)
            else:
                opt_back = optimizer(rate, loss_value)
            return opt_back, loss_back

    def metrics_operation():
        with tf.name_scope('metrics'):
            if type(metric) is str:
                return metricfromname(metric, y_true, y_pred)
            else:
                return metric(y_true, y_pred)

    # reset graph
    tf.reset_default_graph()

    # data
    with tf.name_scope('data'):
        data, traincount, validcount = datafunc()

        if type(data) is dict:
            for key, val in data.items():
                data[key] = val.batch(batch_size)
            dataset = data['train'].concatenate(data['valid'])
        else:
            dataset = data.batch(batch_size)
        dataset = dataset.repeat(epoch)
        iterator = dataset.make_initializable_iterator()
        img, label = iterator.get_next()
        img = tf.image.resize(img, size=reshape, method=reshape_method)

    # arch
    y_pred = arch(img)
    y_true = label
    shape = tf.shape(y_pred)
    y_true = tf.image.resize(y_true, size=[shape[1], shape[2]], method=reshape_method)

    opt, loss_tensor = back_operation()
    metric_tensor = metrics_operation()

    # saver & sess
    if distrainable:
        trainable = [i for i in tf.trainable_variables() if i.name.split('/')[0].split('_')[-1] not in distrainable]

        saver = tf.train.Saver(trainable)
    else:
        saver = tf.train.Saver()
    with tf.Session() as sess:

        # init
        error_rise_count = 0
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        sess.run(iterator.initializer)

        # retrain
        if type(retrain) is dict:
            recorder = retrain
            best_epoch_lst = [j[1] for j in [i.split('_') for i in os.listdir(ckpt_path)] if j[0] == 'epoch']
            assert len(best_epoch_lst) == 1, 'len(best_epoch) == 1'
            best_epoch = int(best_epoch_lst[0])
            start_epoch = best_epoch + 1
            saver.restore(sess=sess, save_path=os.path.join(ckpt_path, f'epoch_{best_epoch}', 'model.ckpt'))
        elif retrain is False:
            mkdir(ckpt_path)
            start_epoch = 1
            recorder = empty_recorder()
            best_epoch = 0
        else:
            assert False, 'Wrong retrain'

        # train & print
        for epoch_num in range(start_epoch, epoch + start_epoch):

            # init
            train_loss, train_count = 0, 0
            valid_metric, valid_count = 0, 0

            # split train valid
            train_chunk = chunk(traincount, batch_size)
            valid_chunk = chunk(validcount, batch_size)

            # tqdm
            if verbose == 2:
                pbar = tqdm(train_chunk)
            else:
                pbar = train_chunk

            for batch in pbar:
                # run training optimizer & get train loss
                try:
                    _, loss_value = sess.run((opt, loss_tensor))
                    train_loss, train_count = cal_mean(
                        train_loss,
                        train_count,
                        loss_value,
                        batch)
                except tf.errors.OutOfRangeError:
                    break
                # valid loss
                if train_chunk.__len__() == 0:
                    for valid_batch in valid_chunk:
                        try:
                            metric_value = sess.run(metric_tensor)
                            valid_metric, valid_count = cal_mean(
                                valid_metric,
                                valid_count,
                                metric_value,
                                valid_batch)
                        except tf.errors.OutOfRangeError:
                            break
                # description
                if verbose == 2:
                    desc_str = f'epoch {epoch_num},' \
                               f' train loss: {round(train_loss, 4)}, ' \
                               f'valid metric: {round(valid_metric, 4)}'
                    pbar.set_description(desc_str)

            recorder['train'].append(train_loss)
            recorder['valid'].append(valid_metric)

            # check point & early stopping
            old_best_epoch = best_epoch
            metrics = recorder['valid']
            new_best_epoch = metrics.index(min(metrics))
            if old_best_epoch != new_best_epoch:
                best_epoch = new_best_epoch
                checkpoint(saver=saver, sess=sess, ckpt_dir=ckpt_path, num_epoch=epoch_num)
                if old_best_epoch != 0:
                    shutil.rmtree(os.path.join(ckpt_path, f'epoch_{old_best_epoch}'))
            else:
                if early_stopping:
                    error_rise_count += 1
                    if error_rise_count >= early_stopping:
                        break

        # final print
        if verbose in [1, 2]:
            print(f'best epoch is {best_epoch}, '
                  f' train score is {recorder["train"][best_epoch]}, '
                  f'valid score is {recorder["valid"][best_epoch]}')

        # write recorder
        recorder['train'] = recorder['train'][: best_epoch + 1]
        recorder['valid'] = recorder['valid'][: best_epoch + 1]

        # write logging
        write_dict = {
            'id': (214013 * int(time.time()) + 2531011) % 2 ** 32,
            'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
            'rate': rate,
            'recorder': recorder,
            'best_epoch': best_epoch,
            'batch_size': batch_size,
            'reshape': reshape,
            'reshape_method': reshape_method
        }
        log = logging()
        log.write(write_dict)
    return recorder


def cv_oof(arch, nflods,
           read_param, gen, train_param, tfr_param, ckpt_dir, verbose=2, retrain=False, rewrite=True):
    # update dir
    if retrain is False:
        mkdir(ckpt_dir)
    # recorder
    recorder = {'info': {}, 'besttrain': [], 'bestvalid': [], 'mean': None, 'std': None}
    # TFR
    if nflods < 10:
        tfr_param['shards'] = nflods * 2
    else:
        tfr_param['shards'] = nflods
    tfr = TFR(**tfr_param)

    # write tfrecords
    if rewrite:
        tfr.write(gen, silence=True)

    for flod in range(nflods):
        # read
        read_param['split'] = nflods
        read_param['valid'] = flod
        read = tfr.read(**read_param)
        # train
        flod_name = f'flod_{flod}'
        if retrain:
            retrain_val = retrain[flod_name]
        else:
            retrain_val = False
        path = os.path.join(ckpt_dir, flod_name)
        train_rcd = train(arch, read, retrain=retrain_val, ckpt_path=path, verbose=verbose, **train_param)
        # recorder
        recorder['besttrain'].append(train_rcd['train'][-1])
        recorder['bestvalid'].append(train_rcd['valid'][-1])
        recorder['info'][flod_name] = train_rcd
    # mean & std
    recorder['mean'] = np.mean(recorder['bestvalid'])
    recorder['std'] = np.std(recorder['bestvalid'])
    # print
    if verbose:
        print(f'Finally,'
              f' mean score is {recorder["mean"]},'
              f' std is {recorder["std"]}')
    return recorder
