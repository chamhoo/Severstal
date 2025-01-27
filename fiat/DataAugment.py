"""
auther: LeeCHH
"""
import tensorflow as tf


def flip_up_down():

    def augment(dataset):
        return dataset.map(lambda img, label: (
            tf.image.random_flip_up_down(img, 18473),
            tf.image.random_flip_up_down(label, 18473)))
    return augment


def flip_left_right():

    def augment(dataset):
        return dataset.map(lambda img, label: (
            tf.image.random_flip_left_right(img, 18473),
            tf.image.random_flip_left_right(label, 18473)))
    return augment


def transpose():
    """
    Transpose image(s) by swapping the height and width dimension.
    """
    def augment(dataset):
        return dataset.map(lambda img, label: (
            tf.image.transpose_image(img),
            tf.image.transpose_image(label)))
    return augment


def brightness(delta):
    """
    - https://www.tensorflow.org/api_docs/python/tf/image/adjust_brightness

    This is a convenience method that converts RGB images to float representation,
    adjusts their brightness, and then converts them back to the original data type.
    If several adjustments are chained, it is advisable to minimize the number of redundant conversions.

    The value delta is added to all components of the tensor image. image is converted to
    float and scaled appropriately if it is in fixed-point representation, and delta is
    converted to the same data type. For regular images, delta should be in the range [0,1),
    as it is added to the image in floating point representation, where pixel values are in the [0,1) range.
    """
    # Constrain delta to [0, 1]
    if delta >= 1:
        delta = 1
    elif delta <= -1:
        delta = -1

    def augment(dataset):
        return dataset.map(lambda img, _: (tf.image.random_brightness(img, delta, 18473), _))
    return augment


def contrast(lower, upper):
    """
    - https://www.tensorflow.org/api_docs/python/tf/image/adjust_contrast

    images is a tensor of at least 3 dimensions . The last 3 dimensions are interpreted
    as [height, width, channels]. The other dimensions only represent a collection of images,
    such as [batch, height, width, channels].

    Contrast is adjusted independently for each channel of each image.

    For each channel, this Op computes the mean of the image pixels in the channel
    and then adjusts each component x of each pixel to (x - mean) * contrast_factor + mean.

    """
    def augment(dataset):
        return dataset.map(lambda img, _: (tf.image.random_contrast(img, lower, upper, 18473), _))
    return augment


def hue(delta):
    """
    - https://www.tensorflow.org/api_docs/python/tf/image/adjust_hue

    image is an RGB image. The image hue is adjusted by converting
    the image(s) to HSV and rotating the hue channel (H) by delta.
    The image is then converted back to RGB.
    delta must be in the interval [-1, 1].
    """
    # Constrain delta to [0, 1]
    if delta >= 1:
        delta = 1
    elif delta <= -1:
        delta = -1

    def augment(dataset):
        return dataset.map(lambda img, _: (tf.image.random_hue(img, delta, 18473), _))
    return augment


def saturation(saturation_factor):
    """
    - https://www.tensorflow.org/api_docs/python/tf/image/adjust_saturation

    image is an RGB image or images. The image saturation is adjusted by converting
    the images to HSV and multiplying the saturation (S) channel by saturation_factor
    and clipping. The images are then converted back to RGB.
    """
    def augment(dataset):
        return dataset.map(lambda img, _:(tf.image.adjust_saturation(img, saturation_factor), _))
    return augment


def standardization():
    """
    - https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization

    This op computes (x - mean) / adjusted_stddev,
    where mean is the average of all values in image,
    and adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements())).

    stddev is the standard deviation of all values in image.
    It is capped away from zero to protect against division by 0 when handling uniform images.
    """
    def augment(dataset):
        return dataset.map(lambda img, _: (tf.image.per_image_standardization(img), _))
    return augment
