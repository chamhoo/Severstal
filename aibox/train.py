"""
auther: leechh
"""
import os
import time
import shutil
import tensorflow as tf
from tqdm import tqdm
from math import ceil

from aibox.model import Model
from aibox.chunk import chunk
from aibox.logging import logging


class Train(Model):
    def empty_modelinfo(self):
        self.modelinfo = {
            'start_epoch': 1,
            'ckpt': '',
            'train_loss': [2**32],
            'valid_metrics': [2**32]
        }

    def __checkpoint(self, saver, sess, ckpt_dir, num_epoch):
        saver.save(sess, os.path.join(ckpt_dir, f'epoch{num_epoch}', 'arch.ckpt'))

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
               again on the basis of a trained arch, if yes,
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
            self.modelinfo['ckpt'] = os.path.join(ckpt_dir, f'epoch{best_epoch}', 'arch.ckpt')
            self.modelinfo['start_epoch'] = best_epoch + 1
            self.modelinfo['train_loss'] = self.modelinfo['train_loss'][: best_epoch + 1]
            self.modelinfo['valid_metrics'] = self.modelinfo['valid_metrics'][: best_epoch + 1]

            # write logging
            write_dict = {
                'id': (214013 * int(time.time()) + 2531011) % 2**32,
                'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                'name': self.model_name,
                'optimizer': self.optimizier_name,
                'rate': self.rate,
                'model_params': self.model_param,
                'ckpt_path': self.modelinfo['ckpt'],
                'loss_name': self.loss_name,
                'loss': self.modelinfo['train_loss'],
                'metric_name': self.metric_name,
                'metric': self.modelinfo['train_loss'],
                'epoch': best_epoch,
                'batch_size': self.batch_size,
                'reshape': self.reshape,
                'reshape_method': self.reshape_method
            }
            log = logging()
            log.write(write_dict)

            # remove unnecessary ckpt
            for ckpt in os.listdir(ckpt_dir):
                if ckpt != f'epoch{best_epoch}':
                    shutil.rmtree(os.path.join(ckpt_dir, ckpt))

