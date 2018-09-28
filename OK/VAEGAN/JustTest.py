#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://github.com/NELSONZHAO/zhihu/tree/master/denoise_auto_encoder

import os
import numpy as np
import tensorflow as tf
print("TensorFlow Version: %s" % tf.__version__)
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', validation_size=0, one_hot=False)
img = mnist.train.images[20]

plt.interactive(False)
plt.imshow(img.reshape((28, 28)))
plt.show()

noise_factor = 0.5


fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
in_imgs = mnist.test.images[10:20]
noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)

reconstructed = noisy_imgs

for images, row in zip([noisy_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)
plt.show()

