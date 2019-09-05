"""
auther: LeeCHH
"""
import numpy as np
import matplotlib.pyplot as plt


def rle2mask(rle, height, width):
    mask = np.zeros(width*height)
    rle = [int(i) for i in rle.strip().split()]
    for start, lengeth in zip(rle[0::2], rle[1::2]):
        mask[start-1: start+lengeth-1] = 1
    return mask.reshape((height, width, 1), order='F')


def mask2rle(mask):
    rle = ''
    lastis0, length = True, 1
    mask = mask.flatten('F')
    for idx, val in enumerate(mask):
        if val != 0:
            if lastis0:
                rle += str(idx+1) + ' '
            else:
                length += 1
            lastis0 = False
        else:
            if not lastis0:
                rle += str(length) + ' '
                length = 1
            lastis0 = True
    return rle.strip()


def plotmask(iminfo, image, figsize):
    """
    :param iminfo: dict, is used to record the ClassId of the image and the corresponding rle information.
     the key of the dict is ClassId, and the value is rle. example: {1: rle1, 2: rle2} or {1: rle1}
    :param image: numpy array, shape [height, width, 3]
    """
    class2rgb = {1: [1, 2], 2: [0, 2], 3: [0, 1], 4: [2]}
    height, width, _ = image.shape
    image_mask = image.astype('float')
    for ClassId, rle in iminfo.items():
        rgblayer = class2rgb[ClassId]
        masklayer = list({0, 1, 2}-set(rgblayer))
        mask = rle2mask(rle, height, width)
        image_mask[:, :, rgblayer] *= np.repeat(1 - mask, len(rgblayer), axis=2)
        img = image_mask[:, :, masklayer]
        msk = np.repeat(mask, len(masklayer), axis=2)
        image_mask[:, :, masklayer] = 255*msk - 2*img*msk + img
    plt.figure(figsize=figsize)
    for i, im in enumerate([image, image_mask]):
        plt.subplot(2, 1, i+1)
        plt.imshow(im.astype('uint8'))
        plt.grid()
        plt.xticks(np.arange(0, width, 50))
        plt.yticks(np.arange(0, height, 50))
    plt.show()

