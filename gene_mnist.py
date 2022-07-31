import argparse
import numpy as np
import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
from PIL import Image as PILImage
import ctypes
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates

import collections
from keras.datasets import mnist, fashion_mnist
import os
from config import hyperparameters
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=32, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(28 - c[1], c[1], -1):
            for w in range(28 - c[1], c[1], -1):
                if np.random.choice([True, False], 1)[0]:
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])
    channels = cv2.filter2D(x, -1, kernel)
    return np.clip(channels, 0, 1) * 255


def motion_blur(x_np, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    x = PILImage.fromarray(x_np)
    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)
    return np.clip(x, 0, 255)  # BGR to RGB


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]
    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:28, :28]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    filename = ['frosts/frost1.png', 'frosts/frost2.png', 'frosts/frost3.png', 'frosts/frost4.jpg', 'frosts/frost5.jpg', 'frosts/frost6.jpg'][idx]
    frost = cv2.imread(filename, 0)
    x_start, y_start = np.random.randint(0, frost.shape[0] - 28), np.random.randint(0, frost.shape[1] - 28)
    frost = frost[x_start:x_start + 28, y_start:y_start + 28]
    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape, loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer, c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.

    x = c[6] * x + (1 - c[6]) * np.maximum(x, x * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer), 0, 1) * 255


def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape, loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0

    m = np.where(liquid_layer > c[3], 1, 0)
    m = gaussian(m.astype(np.float32), sigma=c[4])
    m[m < 0.8] = 0

    # mud spatter
    color = 63 / 255. * np.ones_like(x) * m
    x *= (1 - m)

    return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = np.clip(x + c, 0, 1)

    return x * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.gray2rgb(x)
    x = sk.color.rgb2hsv(x)
    x = np.clip(x * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    x = sk.color.rgb2gray(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x_np, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]
    x = PILImage.fromarray(x_np)
    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)
    return np.array(x)


def pixelate(x_np, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    x = PILImage.fromarray(x_np)
    x = x.resize((int(28 * c), int(28 * c)), PILImage.BOX)
    x = x.resize((28, 28), PILImage.BOX)

    return np.array(x)


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1):
    c = [(28 * 2, 28 * 0.7, 28 * 0.1),
         (28 * 2, 28 * 0.08, 28 * 0.2),
         (28 * 0.05, 28 * 0.01, 28 * 0.02),
         (28 * 0.07, 28 * 0.01, 28 * 0.02),
         (28 * 0.12, 28 * 0.01, 28 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape

    # random affine
    center_square = np.float32(shape) // 2
    square_size = min(shape) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape, borderMode=cv2.BORDER_CONSTANT)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='constant').reshape(shape), 0, 1) * 255


# /////////////// End Distortions ///////////////
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataName', default='mnist', type=str, choices=['mnist', 'fashion'])
    return parser.parse_args()


def main():
    args = get_args()
    dataName = args.dataName
    global x_ori
    parameters = hyperparameters(dataName, None)
    d = collections.OrderedDict()
    d['Gaussian_Noise'] = gaussian_noise
    d['Shot_Noise'] = shot_noise
    d['Impulse_Noise'] = impulse_noise
    d['Defocus_Blur'] = defocus_blur
    d['Glass_Blur'] = glass_blur
    d['Motion_Blur'] = motion_blur
    d['Zoom_Blur'] = zoom_blur
    d['Snow'] = snow
    d['Frost'] = frost
    d['Fog'] = fog
    d['Brightness'] = brightness
    d['Contrast'] = contrast
    d['Pixelate'] = pixelate
    d['JPEG'] = jpeg_compression
    d['Elastic'] = elastic_transform

    if dataName == "fashion":
        (_, _), (x_ori, _) = fashion_mnist.load_data()
    elif dataName == "mnist":
        (_, _), (x_ori, _) = mnist.load_data()

    for method_name in d.keys():
        print('Creating images for the corruption', method_name)
        x_new = np.zeros_like(x_ori)
        for severity in range(1, 6):
            corruption = lambda clean_img: d[method_name](clean_img, severity)
            for imgNo, img in enumerate(x_ori):
                img_new = corruption(img)
                x_new[imgNo] = img_new
            print("{0}-{1}".format(method_name, severity))
            np.save(parameters.save_data_root_adv + "{0}-{1}.npy".format(method_name, severity), x_new)
            print("{0}-{1}: {2}".format(x_new.shape, x_new.min(), x_new.max()))


if __name__ == '__main__':
    main()