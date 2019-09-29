"""
auther: leechh
"""
import os
import time
import shutil
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from math import ceil
from tools.model import Model
from tools.preprocess import Preprocess
from tools.chunk import chunk


class Network(Model, Preprocess):

    def logging(self, path=None):
        # mk
        if path is None:
            path = 'training.log'

        if not os.path.exists(path):
            with open(path, 'w') as file:
                col_str = 'time,name,' \
                          'rate,model_params,ckpt_path,' \
                          'loss_name,loss,metric_name,metric,' \
                          'epoch,batch_size,reshape,reshape_method\n'
                file.write(col_str)
                print('ok')

        with open(path, 'a+') as file:
            log_str = f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))},' \
                      f'{self.model_name},' \
                      f'{self.rate},' \
                      f'{self.model_param},' \
                      f'{self.modelinfo["ckpt"]},' \
                      f'{self.loss_name},' \
                      f'{self.modelinfo["train_loss"][-1]},' \
                      f'{self.metric_name},' \
                      f'{self.modelinfo["valid_metrics"][-1]},' \
                      f'{self.modelinfo["start_epoch"]-1},' \
                      f'{self.batch_size},' \
                      f'{self.reshape},' \
                      f'{self.reshape_method}\n'
            file.write(log_str)

    def select_model(self, model):
        if model == 'unet':
            return self.unet
        else:
            assert False, 'model ISNOT exist'

    def model(self, model_name, model_params,
              loss, metric, optimizer, rate):

        # parameter
        self.rate = rate
        self.loss_name = loss
        self.metric_name = metric
        self.optimizier_name = optimizer
        self.model_name = model_name
        self.model_param = model_params

        # select model
        model_params['height'], model_params['width'] = self.reshape
        model = self.select_model(model_name)

        # train
        y_pred, y_h, y_w = model(x=self.img, **model_params)

        y_true = tf.image.resize(
            self.label,
            size=[y_h, y_w],
            method=self.reshape_method)

        self.loss = self.metric_func(
            metric_name=self.loss_name,
            y_true=y_true,
            y_pred=y_pred)

        self.metric = self.metric_func(
            metric_name=self.metric_name,
            y_true=y_true,
            y_pred=y_pred)

        self.opt = self.optimizier(
            optimizier_name=optimizer,
            learning_rate=rate,
            loss=self.loss)

    def empty_modelinfo(self):
        self.modelinfo = {
            'start_epoch': 1,
            'ckpt': '',
            'train_loss': [2**32],
            'valid_metrics': [2**32]
        }

    def __checkpoint(self, saver, sess, ckpt_dir, num_epoch):
        saver.save(sess, os.path.join(ckpt_dir, f'epoch{num_epoch}', 'model.ckpt'))

    def cal_mean(self, oldmean, oldcount, mean, count):
        newcount = count + oldcount
        newmean = (oldcount * oldmean) + (mean * count)
        return newmean / newcount, newcount

    def train(self, ckpt_dir, train_percentage=0.8, early_stopping=5, verbose=2, retrain=False):
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
        # parameter
        assert (train_percentage > 0) & (train_percentage < 1), 'train_percentage is out of range (0, 1)'
        assert (early_stopping >= 0) & (type(early_stopping) is int), 'early_stopping > 0 & type is int'
        assert verbose in [0, 1, 2], 'verbose is out of range [0, 1, 2]'
        assert (retrain == 1) or (retrain == 0), 'retrain is out of range [0, 1], No matter what type'

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
            sess.run(self.iterator.initializer)

            # train & print
            start_epoch = self.modelinfo['start_epoch']
            for epoch_num in range(start_epoch, self.epoch + start_epoch):

                # init
                best_epoch = 0
                train_loss, train_count = 0, 0
                valid_metric, valid_count = 0, 0

                # split train valid
                data_chunk = chunk(self.traincount, self.batch_size)
                num_train_chunk = ceil(train_percentage * data_chunk.__len__())
                train_chunk = data_chunk.take(num_train_chunk)

                # tqdm
                if verbose == 2:
                    pbar = tqdm(train_chunk)
                else:
                    pbar = train_chunk

                for batch in pbar:
                    # run training optimizer & get train loss
                    try:
                        _, loss = sess.run((self.opt, self.loss))
                        train_loss, train_count = self.cal_mean(
                            train_loss,
                            train_count,
                            loss,
                            batch)
                    except tf.errors.OutOfRangeError:
                        break
                    # valid loss
                    if train_chunk.__len__() == 0:
                        for valid_batch in data_chunk:
                            try:
                                metric = sess.run(self.metric)
                                valid_metric, valid_count = self.cal_mean(
                                    valid_metric,
                                    valid_count,
                                    metric,
                                    valid_batch)
                            except tf.errors.OutOfRangeError:
                                break
                    # description
                    if verbose == 2:
                        desc_str = f'epoch {epoch_num},' \
                                  f' train {self.loss_name}: {round(train_loss, 4)}, ' \
                                  f'valid {self.metric_name}: {round(valid_metric, 4)}'
                        pbar.set_description(desc_str)

                self.modelinfo['train_loss'].append(train_loss)
                self.modelinfo['valid_metrics'].append(valid_metric)

                # check point & early stopping
                old_best_epoch = best_epoch
                if early_stopping:
                    metrics = self.modelinfo['valid_metrics']
                    best_epoch = metrics.index(min(metrics))
                    if best_epoch != epoch_num:
                        error_rise_count += 1
                        if error_rise_count >= early_stopping:
                            break
                else:
                    best_epoch = epoch_num

                if old_best_epoch != best_epoch:
                    if old_best_epoch != 0:
                        shutil.rmtree(os.path.join(ckpt_dir, f'epoch{old_best_epoch}'))
                    self.__checkpoint(saver=saver,
                                      sess=sess,
                                      ckpt_dir=ckpt_dir,
                                      num_epoch=epoch_num)
                    
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

            self.logging()

            # remove unnecessary ckpt
            for ckpt in os.listdir(ckpt_dir):
                if ckpt != f'epoch{best_epoch}':
                    shutil.rmtree(os.path.join(ckpt_dir, ckpt))

    def cv(self, readtrain_params, model_params, keepckpt=True, ckpt_dir=None,
           nfolds=5, train_percentage=0.8, early_stopping=False, verbose=2):

        # parameter
        assert type(keepckpt) is bool, 'the type of keepckpt is bool'
        assert (train_percentage > 0) & (train_percentage < 1), 'train_percentage is out of range (0, 1)'
        assert (early_stopping >= 0) & (type(early_stopping) is int), 'early_stopping > 0 & type is int'
        assert verbose in [0, 1, 2], 'verbose is out of range [0, 1, 2]'
        assert (nfolds >= 0) & (type(nfolds) is int), 'nfolds > 0 & type is int'

        if ckpt_dir is None:
            ckpt_dir = 'cv'

        # init
        self.mkdir(ckpt_dir)
        score_list = []
        result = {'ckpt': [], 'train_loss': {}, 'valid_metrics': {}}

        # train parameter
        train_params = {}
        if verbose == 2:
            train_params['verbose'] = 2
        else:
            train_params['verbose'] = 0
        train_params['retrain'] = False
        train_params['train_percentage'] = train_percentage
        train_params['early_stopping'] = early_stopping

        # train
        for fold in range(1, 1 + nfolds):
            train_params['ckpt_dir'] = os.path.join(ckpt_dir, f'fold{fold}')
            # reset seed & train
            self.__seed = next(self.random_gen)
            self.readtrain(**readtrain_params)
            self.model(**model_params)
            self.train(**train_params)
            # get score
            best_epoch = self.modelinfo['start_epoch'] - 1
            best_score = self.modelinfo['valid_metrics'][best_epoch]
            score_list.append(best_score)
            # filling result
            result['train_loss'][f'fold{fold}'] = self.modelinfo['train_loss']
            result['valid_metrics'][f'fold{fold}'] = self.modelinfo['valid_metrics']
            result['ckpt'].append(self.modelinfo['ckpt'])
            # print if verbose is 1
            if verbose == 1:
                print(f'fold {fold}, score is {round(best_score, 4)}')

        # drop ckpt if keepckpt is False
        if keepckpt is False:
            shutil.rmtree(ckpt_dir)
            result.__delitem__('ckpt')

        # mean & std
        result['mean_score'] = np.mean(score_list)
        result['std_score'] = np.std(score_list)

        if verbose:
            print(f'Finally,'
                  f' mean score is {result["mean_score"]},'
                  f' std is {result["mean_std"]}')

        return result

