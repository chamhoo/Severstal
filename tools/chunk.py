from math import ceil


class chunk(object):

    def __init__(self, total, batch):
        assert (total > 0) & (batch > 0), 'total and batch must be greater than 0.'
        assert ((type(total) is int) & (type(batch) is int)), 'The type of total and batch must be int.'

        self.total = total
        self.batch = batch

    def __len__(self):
        return ceil(self.total / self.batch)

    def __next__(self):
        if self.total > 0:
            old_total = self.total
            self.total -= self.batch
            if self.batch <= old_total:
                return self.batch
            else:
                return old_total
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def take(self, num):
        assert type(num) is int, 'the type of num is int.'
        assert num > 0, 'num > 0'
        assert num < self.__len__(), 'num is too big'

        for _ in range(num):
            self.total -= self.batch
        return chunk(num * self.batch, self.batch)