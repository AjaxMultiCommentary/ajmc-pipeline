import numpy as np
from skimage import color
from skimage.util import random_noise
from skimage.filters import threshold_otsu, butterworth
from skimage.morphology import binary_closing, binary_opening, binary_erosion, binary_dilation, skeletonize
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats, clear_output
from sympy import N
set_matplotlib_formats('svg')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def align_rgb_values(img):
    # input is numpy array
    mean = np.mean(img, axis=2, keepdims=True)
    mean_img = np.tile(mean, (1,1,3))
    return np.array(mean_img, dtype='uint8')

def preprocess_img(img, instructions=[]):
    steps = []

    steps.append(["original", img])

    def cross_kernel(size):
        assert size % 2 == 1
        tmp_kernel = np.zeros((size, size))
        half = (size-1) // 2
        tmp_kernel[:,half] = 1
        tmp_kernel[half,:] = 1
        return tmp_kernel

    # 3x3 kernel
    kernel_3x3 = np.ones((3,3))

    # rgb to greyscale
    tmp_img = color.rgb2gray(img)
    steps.append(["rgb2grey", tmp_img])

    # binarization
    threshold = threshold_otsu(tmp_img)
    tmp_img = tmp_img > threshold
    steps.append(["otsu", tmp_img])
   
    # erosion
    tmp_img = binary_erosion(tmp_img, footprint=np.ones((3,3)))
    steps.append(["erosion", tmp_img])

    # opening
    tmp_img = binary_opening(tmp_img, footprint=cross_kernel(3))
    steps.append(["opening", tmp_img])

    # skeletonize
    # tmp_img = 1-skeletonize(1-tmp_img, method='lee')
    # steps.append(["skeletonize", tmp_img])

    # closing
    # tmp_img = binary_closing(tmp_img, footprint=cross_kernel(5))
    # steps.append(["closing", tmp_img])

    # dilation
    # tmp_img = binary_dilation(tmp_img, footprint=cross_kernel(3))
    # steps.append(["dilation", tmp_img])

    return steps

def add_noise(img, noise_type, show_fig=True):
    if noise_type.lower() in ["s&p"]:
        # Add salt-and-pepper noise to the image.
        noise_img = random_noise(img, mode='s&p',amount=0.4)
    elif noise_type.lower() in ["gaussian"]:
        noise_img = random_noise(img, mode='gaussian', clip=True, mean=0, var=0.2)
    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = align_rgb_values(255*noise_img)
    if show_fig:
        plt.imshow(noise_img)
        plt.axis("off")
    return noise_img