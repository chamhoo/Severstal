import tensorflow as tf


def decode_img_seg(raw):
    img = tf.decode_raw(raw['img'], tf.uint8)
    label = tf.decode_raw(raw['label'], tf.bool)
    # convert image dtype
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    label = tf.image.convert_image_dtype(label, dtype=tf.float32)
    return img, label

