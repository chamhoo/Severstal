"""
auther: leechh
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tools.model import Model
from tools.preprocess import Preprocess
from tools.mask import plotgen


class Network(Model, Preprocess):

    def select_model(self, model):
        if model == 'unet':
            return self.unet
        else:
            assert False, 'model ISNOT exist'

    def test(self):
        with tf.Session() as sess:
            # init
            sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
            sess.run((self.train_iterator.initializer, self.valid_iterator.initializer))
            for _ in range(5):
                try:
                    img, label = sess.run((
                        self.train_img,
                        tf.image.resize(self.train_label, size=[128, 800], method=3)))
                    plotgen(label[0], img[0], [15, 7])
                except tf.errors.OutOfRangeError:
                    break

    def model(self, model_name, model_params,
              loss, metric, optimizer, rate):

        # parameter
        self.loss_name = loss
        self.metric_name = metric

        # select model
        model_params['height'], model_params['width'] = self.reshape
        model = self.select_model(model_name)

        # train
        train_y_pred, y_h, y_w = model(x=self.train_img, **model_params)

        train_y_true = tf.image.resize(
            self.train_label,
            size=[y_h, y_w],
            method=self.reshape_method)

        self.loss = self.metric_func(
            metric_name=self.loss_name,
            y_true=train_y_true,
            y_pred=train_y_pred)

        self.opt = self.optimizier(
            optimizier_name=optimizer,
            learning_rate=rate,
            loss=self.loss[0])

        # valid
        valid_y_pred, y_h, y_w = model(x=self.valid_img, **model_params)

        valid_y_true = tf.image.resize(
            self.valid_label,
            size=[y_h, y_w],
            method=self.reshape_method)

        self.metric = self.metric_func(
            metric_name=self.metric_name,
            y_true=valid_y_true,
            y_pred=valid_y_pred)

    def empty_modelinfo(self):
        self.modelinfo = {
            'start_epoch': 1,
            'ckpt': None,
            'train_loss': [2**32],
            'valid_metrics': [2**32]
        }

    def __checkpoint(self, saver, sess, ckpt_dir, num_epoch):
        saver.save(sess, os.path.join(ckpt_dir, f'epoch{num_epoch}', 'model.ckpt'))

    def train(self, ckpt_dir, early_stopping=5, verbose=2, retrain=False):
        """

        :param ckpt_dir: ckpt file storage directory.
        :param early_stopping: will stop training if the metric of valid data
               doesn't improve in last {early_stopping} epoch.
               if 0(False),it mean don't early_stopping
        :param verbose: 0, 1, 2, Print information level.
               0=silent, 1=just finally score, 2=progress bar.
        :param retrain: bool, default False, Whether to train
               again on the basis of a trained model, if yes,
               please select True, otherwise False.
        """
        # saver & sess
        saver = tf.train.Saver()
        with tf.Session() as sess:

            # retrain
            if retrain is True:
                saver.restore(sess=sess, save_path=self.modelinfo['ckpt'])
            else:
                self.empty_modelinfo()
            # init
            error_rise_count = 0
            self.mkdir(ckpt_dir)
            sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
            sess.run((self.train_iterator.initializer, self.valid_iterator.initializer))

            # train & print
            start_epoch = self.modelinfo['start_epoch']
            for epoch_num in range(start_epoch, self.epoch + start_epoch):

                # init
                loss_epoch = []
                metric_epoch = []

                num_train_chunk = int(np.ceil(self.train_count / self.batch_size))
                num_valid_chunk = int(np.ceil(self.valid_count / self.batch_size))

                # run training optimizer
                for _ in range(num_train_chunk):
                    try:
                        sess.run(self.opt)
                    except tf.errors.OutOfRangeError:
                        break

                # get training loss
                for _ in range(num_train_chunk):
                    try:
                        loss_epoch.append(sess.run(self.loss))
                    except tf.errors.OutOfRangeError:
                        break

                # get valid loss
                for _ in range(num_valid_chunk):
                    try:
                        metric_epoch.append(sess.run(self.metric))
                    except tf.errors.OutOfRangeError:
                        break

                # calculate loss and print
                loss_epoch = sum(loss_epoch)
                metric_epoch = sum(metric_epoch)

                self.modelinfo['train_loss'].append(loss_epoch)
                self.modelinfo['valid_metrics'].append(metric_epoch)

                if verbose:
                    print(f'After {epoch_num} epoch, '
                          f'train {self.loss_name} is {round(loss_epoch, 4)}, '
                          f'valid {self.metric_name} is {round(metric_epoch, 4)}')

                # check point & early stopping
                self.__checkpoint(saver=saver,
                                  sess=sess,
                                  ckpt_dir=ckpt_dir,
                                  num_epoch=epoch_num)

                if early_stopping:
                    metrics = self.modelinfo['valid_metrics']
                    best_epoch = metrics.index(min(metrics))
                    if best_epoch != epoch_num:
                        error_rise_count += 1
                        if error_rise_count >= early_stopping:
                            break
                else:
                    best_epoch = epoch_num

            # final print
            if verbose in [1, 2]:
                print(f'best epoch is {best_epoch}, '
                      f' train score is {self.modelinfo["train_loss"][best_epoch]}, '
                      f'valid score is {self.modelinfo["valid_metrics"][best_epoch]}')

            # write modelinfo
            self.modelinfo['ckpt'] = os.path.join(ckpt_dir, f'epoch{best_epoch}', 'model.ckpt')
            self.modelinfo['start_epoch'] = best_epoch + 1
            self.modelinfo['train_loss'] = self.modelinfo['train_loss'][: best_epoch + 1]
            self.modelinfo['valid_metrics'] = self.modelinfo['valid_metrics'][: best_epoch + 1]