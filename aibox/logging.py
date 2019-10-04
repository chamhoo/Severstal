"""
auther: leechh
"""
import os
import time
from functools import reduce


class logging(object):
    def __init__(self, path=None):
        self.log_col = 'id;time;name;optimizer;rate;' \
                  'model_params;ckpt_path;' \
                  'loss_name;loss;metric_name;metric;' \
                  'epoch;batch_size;reshape;reshape_method'

        if path is None:
            self.path = ['training.log']
        else:
            if type(path) is str:
                self.path = list(path)
            else:
                self.path = path

        # mkfile if not exist
        if not os.path.exists(self.path[0]):
            with open(self.path[0], 'w') as file:
                file.write(self.log_col + '\n')

    def __len__(self):
        return reduce(lambda x, y: x + len(open(y).readlines()), self.path)

    def __add__(self, other):
        return logging(path=self.path + other.path)

    def __iadd__(self, other):
        self.path += other.path

    def gen(self):
        for path in self.path:
            with open(path) as file:
                idx = 0
                for line in file:
                    line = line.strip().split(';')
                    yield line, idx
                    idx += 1

    def __contains__(self, item):
        position = 0
        for line, idx in self.gen():
            if idx == 0:
                position = line.index('id')
            else:
                if item == line[position]:
                    return True
        return False

    def __getitem__(self, item):
      pass

    def write(self, write_dict):
        # write log
        for path in self.path:
            log_str = ''
            with open(path, 'a+') as file:
                for col in self.log_col.strip().split(';'):
                    try:
                        log_str += str(write_dict[col])
                    except KeyError:
                        log_str += ''
                    finally:
                        log_str += ';'
                file.write(log_str[:-1] + '\n')

    def localized(self, path):
        assert not os.path.exists(path), 'path is exist.'
        # mkfile
        with open(path, 'w') as file:
            file.write(self.log_col + '\n')

        col = []
        for line, idx in self.gen():
            if idx == 0:
                col = line
            else:
                write_dict = dict(zip(col, line))
                self.write(write_dict)