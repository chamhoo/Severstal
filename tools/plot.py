"""
auther: LeeCHH
"""
import numpy as np
import matplotlib.pyplot as plt


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


def plotgen(iminfo, image, figsize):
    class2rgb = {1: [1, 2], 2: [0, 2], 3: [0, 1], 4: [2]}
    # image = np.frombuffer(image, dtype='uint8').reshape([256, 1600, 3])
    # iminfo = np.frombuffer(iminfo, dtype='uint8').reshape([256, 1600, 4])
    height, width, _ = image.shape
    image_mask = image.astype('float')

    for layer in range(4):
        mask = iminfo[..., layer][..., np.newaxis]
        rgblayer = class2rgb[layer + 1]
        masklayer = list({0, 1, 2} - set(rgblayer))
        image_mask[:, :, rgblayer] *= np.repeat(1 - mask, len(rgblayer), axis=2)
        img = image_mask[:, :, masklayer]
        msk = np.repeat(mask, len(masklayer), axis=2)
        image_mask[:, :, masklayer] = 255 * msk - 2 * img * msk + img

    plt.figure(figsize=figsize)
    for i, im in enumerate([image, image_mask]):
        plt.subplot(2, 1, i + 1)
        plt.imshow(im.astype('uint8'))
        plt.grid()
        plt.xticks(np.arange(0, width, 50))
        plt.yticks(np.arange(0, height, 50))
    plt.show()