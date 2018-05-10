# -*- coding: utf-8 -*-
'''
@author:
BinCai (bincai@tencent.com)
https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
'''

"""
Imports
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import urllib.request

def ptb_iterator(raw_data, batch_size, num_steps, steps_ahead=1):
  """Iterate on the raw PTB data.
  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.
  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)

  batch_len = data_len // batch_size

  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  offset = 0
  if data_len % batch_size:
    offset = np.random.randint(0, data_len % batch_size)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i + offset:batch_len * (i + 1) + offset]

  epoch_size = (batch_len - steps_ahead) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+steps_ahead]
    yield (x, y)

  if epoch_size * num_steps < batch_len - steps_ahead:
    yield (data[:, epoch_size*num_steps : batch_len - steps_ahead], data[:, epoch_size*num_steps + 1:])

def shuffled_ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)
    r = len(raw_data) % num_steps
    if r:
        n = np.random.randint(0, r)
        raw_data = raw_data[n:n + len(raw_data) - r]

    raw_data = np.reshape(raw_data, [-1, num_steps])
    np.random.shuffle(raw_data)

    num_batches = int(np.ceil(len(raw_data) / batch_size))

    for i in range(num_batches):
        data = raw_data[i*batch_size:min(len(raw_data), (i+1)*batch_size),:]
        yield (data[:,:-1], data[:,1:])

file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
file_name = 'tinyshakespeare.txt'
if not os.path.exists(file_name):
    urllib.request.urlretrieve(file_url, file_name)

with open(file_name,'r') as f:
    raw_data = f.read()
    print("Data length:", len(raw_data))

vocab = set(raw_data)
vocab_size = len(vocab)
idx_to_vocab = dict(enumerate(vocab))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

data = [vocab_to_idx[c] for c in raw_data]
del raw_data

def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield ptb_iterator(data, batch_size, num_steps)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def train_network(g, num_epochs, num_steps = 200, batch_size = 32, verbose = True, save=False):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1

                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                      g['final_state'],
                                                      g['train_step']],
                                                             feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses

