"""
auther: leechh
"""
import os


def empty_recorder():
    return {'train': [-(2 ** 32)], 'valid': [-(2 ** 32)]}


def checkpoint(saver, sess, ckpt_dir, num_epoch):
    saver.save(
        sess=sess,
        save_path=os.path.join(ckpt_dir, f'epoch_{num_epoch}', 'model.ckpt'),
        write_meta_graph=False)


def cal_mean(oldmean, oldcount, mean, count):
    newcount = count + oldcount
    newmean = (oldcount * oldmean) + (mean * count)
    return newmean / newcount, newcount