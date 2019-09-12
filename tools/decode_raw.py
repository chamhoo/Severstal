import tensorflow as tf


def decode_img_seg(raw):
    img = tf.decode_raw(raw['img'], tf.float32)
    label = tf.decode_raw(raw['label'], tf.float32)
    return img, label

