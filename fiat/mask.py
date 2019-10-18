"""
auther: LeeCHH
"""
import numpy as np


def rle2mask(rle, height, width):
    mask = np.zeros(width*height, dtype='bool')
    rle = [int(i) for i in rle.strip().split()]
    for start, lengeth in zip(rle[0::2], rle[1::2]):
        mask[start-1: start+lengeth-1] = True
    return mask.reshape((height, width, 1), order='F')


def mask2rle(img):
    """
    - https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    """
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


