import tensorflow as tf


def decode_img_seg(raw):
    img = tf.decode_raw(raw['img'], tf.uint8)
    label = tf.decode_raw(raw['label'], tf.uint8)

    # convert image dtype
    img = tf.cast(img, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.float32)

    # set_shape
    img = tf.reshape(img, [raw['height'], raw['width'], raw['channels']])
    label = tf.reshape(label, [raw['height'], raw['width'], raw['n_class']])
    return img, label

