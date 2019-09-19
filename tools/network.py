"""
auther: leechh
"""
import numpy as np
import tensorflow as tf
from tools.model import Model
from tools.preprocess import Preprocess


class Network(Model, Preprocess):

    def select_model(self, model):
        if model == 'unet':
            return self.unet
        else:
            assert False, 'model ISNOT exist'

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
            loss=self.loss_value)

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

    def train(self, early_stopping, verbose, retrain):
        # train & metric
        with tf.Session() as sess:
            # init
            sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
            sess.run((self.train_iterator.initializer, self.valid_iterator.initializer))

            # train & print
            for epoch_num in range(1, self.epoch+1):

                # init
                loss_epoch = []
                metric_epoch = []

                num_train_chunk = int(np.ceil(self.train_count / self.batch_size))
                num_valid_chunk = int(np.ceil(self.valid_count / self.batch_size))

                # run training optimizer
                for _ in range(num_train_chunk):
                    sess.run(self.opt)

                # get training loss
                for _ in range(num_train_chunk):
                    loss_epoch.append(sess.run(self.loss))

                # get valid loss
                for _ in range(num_valid_chunk):
                    metric_epoch.append(sess.run(self.metric))

                # calculate loss and print
                loss_epoch = sum(loss_epoch)
                metric_epoch = sum(metric_epoch)

                if verbose:
                    print(f'After {epoch_num} epoch,'
                          f' train {self.loss_name} is {loss_epoch}, '
                          f'valid {self.metric_name} is {metric_epoch}')