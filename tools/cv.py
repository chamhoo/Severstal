"""
auther: leechh
"""
import os
import shutil
import numpy as np
from tools.train import Train


class CV(Train):

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