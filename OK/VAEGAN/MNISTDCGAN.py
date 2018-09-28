#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://github.com/NELSONZHAO/zhihu/tree/master/denoise_auto_encoder

import os
import numpy as np
import tensorflow as tf
print("TensorFlow Version: %s" % tf.__version__)
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#   ---------------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

mnist = input_data.read_data_sets('MNIST_data', validation_size=0, one_hot=False)
img = mnist.train.images[20]

plt.interactive(False)
plt.imshow(img.reshape((28, 28)))
plt.show()

inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs_')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets_')

#Encoder 三层卷积
conv1 = tf.layers.conv2d(inputs_, 64, (3,3), padding='same', activation=tf.nn.relu)
#Tensor("conv2d/Relu:0", shape=(?, 28, 28, 64), dtype=float32)
conv1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
#Tensor("max_pooling2d/MaxPool:0", shape=(?, 14, 14, 64), dtype=float32)

conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=tf.nn.relu)
#Tensor("conv2d_2/Relu:0", shape=(?, 14, 14, 64), dtype=float32)
conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
#Tensor("max_pooling2d_2/MaxPool:0", shape=(?, 7, 7, 64), dtype=float32)

conv3 = tf.layers.conv2d(conv2, 32, (3,3), padding='same', activation=tf.nn.relu)
#Tensor("conv2d_3/Relu:0", shape=(?, 7, 7, 32), dtype=float32)
conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
#Tensor("max_pooling2d_3/MaxPool:0", shape=(?, 4, 4, 32), dtype=float32)

#Decoder 三层卷积
conv4 = tf.image.resize_nearest_neighbor(conv3, (7,7)) #通过临近插值法将 `images`大小调整至`size`，输出为float类型
#Tensor("ResizeNearestNeighbor:0", shape=(?, 7, 7, 32), dtype=float32)
conv4 = tf.layers.conv2d(conv4, 32, (3,3), padding='same', activation=tf.nn.relu)
#Tensor("conv2d_4/Relu:0", shape=(?, 7, 7, 32), dtype=float32)

conv5 = tf.image.resize_nearest_neighbor(conv4, (14,14))
#Tensor("ResizeNearestNeighbor_1:0", shape=(?, 14, 14, 32), dtype=float32)
conv5 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=tf.nn.relu)
#Tensor("conv2d_5/Relu:0", shape=(?, 14, 14, 64), dtype=float32)

conv6 = tf.image.resize_nearest_neighbor(conv5, (28,28))
#Tensor("ResizeNearestNeighbor_2:0", shape=(?, 28, 28, 64), dtype=float32)
conv6 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=tf.nn.relu)
#Tensor("conv2d_6/Relu:0", shape=(?, 28, 28, 64), dtype=float32)

#logits and outputs
logits_ = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)
#Tensor("conv2d_7/BiasAdd:0", shape=(?, 28, 28, 1), dtype=float32)
outputs_ = tf.nn.sigmoid(logits_, name='outputs_')

#loss and Optimizer
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
merged_summary_op = tf.summary.merge_all()

modeldir='./model.ae'
if not os.path.exists(modeldir): os.mkdir(modeldir)

#训练
sess = tf.Session()
noise_factor = 0.5
epochs = 100
batch_size = 128
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(modeldir, graph=tf.Graph()) #graph=tf.get_default_graph())

for e in range(epochs):
  for idx in range(mnist.train.num_examples // batch_size):
    batch = mnist.train.next_batch(batch_size)
    imgs = batch[0].reshape((-1, 28, 28, 1))

    # 加入噪声
    noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)
    batch_cost, summary, _ = sess.run([cost, merged_summary_op, optimizer], feed_dict={inputs_: noisy_imgs, targets_: imgs})

    summary_writer.add_summary(summary, e*epochs+idx)
    print("Epoch: {}/{} ".format(e + 1, epochs), "Training loss: {:.4f}".format(batch_cost))

summary_writer.close()

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
in_imgs = mnist.test.images[10:20]
noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)

reconstructed = sess.run(outputs_, feed_dict={inputs_: noisy_imgs.reshape((10, 28, 28, 1))})
	
saver = tf.train.Saver()
saver.save(sess, os.path.join(modeldir, "model.ckpt"))
	
sess.close()

for images, row in zip([noisy_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)
plt.show()

