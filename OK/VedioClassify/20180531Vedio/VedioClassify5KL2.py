#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
from VedioClassifyInputAnsy import VedioClassifyInputAnsy

param = {
  'inputpath': 'data/',
  'modelpath': 'model/',
  'dataset': ['cdata', 'cdata', 'cdata'],
  'testset': ['cdata', 'cdata'],
  'predset': [],

  'shuffle_file': True,
  'batch_size': 32,
  'batch_size_test': 1024,
  'test_batch': 100,
  'save_batch': 500,
  'total_batch': 1000,
  'decay_steps': 1000,
  'keep_prob': 0.5
}

param2 = {
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/classify/data/',
  'modelpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/classify/model/',

  'dataset': ['train0', 'train1', 'train2', 'train3', 'train4',
              'train5', 'train6', 'train7', 'train8', 'train9', 'train10'],
  'testset': ['test0', 'test1'],
  'predset': [],

  'shuffle_file': True,
  'batch_size': 64, # 12862
  'batch_size_test': 10240,
  'test_batch': 1000,
  'save_batch': 6000,
  'total_batch': 1000000,
  'decay_steps': 5000,
  'keep_prob': 0.5
}

#param.update(param2)

def main():
  start_time = time.strftime('%m%d%Y%H%M', time.localtime(time.time()))
  print(param)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  input_dim = 1000
  input_dim2 = 2000
  input_dim5 = 5000
  output_dim = 28
  output_dim2 = 174

  ##----------------------------input
  with tf.name_scope('input') as scope:
    lda1000 = tf.placeholder(dtype='float', shape=[None, input_dim], name='input_lda1000')
    lda2000 = tf.placeholder(dtype='float', shape=[None, input_dim2], name='input_lda2000')
    lda5000 = tf.placeholder(dtype='float', shape=[None, input_dim5], name='input_lda2000')
    label2 = tf.placeholder(dtype='float', shape=[None, output_dim2], name='input_label')

  with tf.name_scope('param') as scope:
    keep_prob = tf.placeholder(dtype="float", name='keep_prob')
    global_step = tf.placeholder(dtype=np.int32, name="global_step")
    
  ##----------------------------concat layer
  with tf.name_scope('concat') as scope:
    concat_item = tf.concat([lda1000, lda2000, lda5000], 1)

  ##----------------------------fc layer
  with tf.name_scope('fc') as scope:
    #fc_w0, fc_b0 = VedioMatchUtils.create_w_b(input_dim, output_dim, w_name="fc_w0", b_name="fc_b0")
    fc_w0 = tf.Variable(tf.zeros([input_dim+input_dim2+input_dim5, output_dim2]), name='fc_w0')  
    fc_b0 = tf.Variable(tf.zeros([output_dim2]), name='fc_b0') 

  ##----------------------------loss layer
  with tf.name_scope('loss') as scope:
    pred = tf.nn.softmax( tf.matmul(concat_item, fc_w0) + fc_b0 )
    cross_entropy = -tf.reduce_sum(label2 * tf.log(pred))
    learning_rate = tf.train.exponential_decay(0.01, global_step, param['decay_steps'], 0.98)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    #optimizer = tf.train.FtrlOptimizer(learning_rate, l1_regularization_strength=0.01, l2_regularization_strength=0.00001)

  ##----------------------------acc compute
  with tf.name_scope('acc') as scope:
    pred_argmax = tf.argmax(pred, 1)
    label_argmax = tf.argmax(label2, 1)
    correct_prediction = tf.equal(pred_argmax, label_argmax)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  readdata = VedioClassifyInputAnsy(param)
  readdata.start_ansyc()
  
  with tf.Session(config=config) as session:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    step = 0
  
    for step in range(param['total_batch']):
      train_data = readdata.read_traindata_batch_ansyc()
  
      ## feed data to tf session
      feed_dict = {
        label2: train_data['label2'],
        lda1000: train_data['lda1000'],
        lda2000: train_data['lda2000'],
        lda5000: train_data['lda5000'],
        global_step: step
      }
  
      if (step > 0 and step % param['test_batch'] == 0):
        ## run optimizer
        _, l, lr, pa, la, acc = session.run([optimizer, cross_entropy, learning_rate, pred_argmax, label_argmax, accuracy], feed_dict=feed_dict)
        print('[Train]\tIter:%d\tloss=%.6f\tlr=%.6f\taccuracy=%.6f\tts=%s' % 
          ( step, l/train_data['L'], lr, acc, time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))), end='\n')
        print('[Train L]\t%s' % str(la[:16]) )
        print('[Train P]\t%s' % str(pa[:16]) )
  
        test_data = readdata.read_testdata_batch_ansyc()
        feed_dict = {
          label2: test_data['label2'],
          lda1000: test_data['lda1000'],
          lda2000: test_data['lda2000'],
          lda5000: test_data['lda5000'],
          global_step: step
        }
  
        ## get acc
        acc, tl, pa, la = session.run([accuracy, cross_entropy, pred_argmax, label_argmax], feed_dict=feed_dict)
  
        print('[Test]\tIter:%d\tloss=%.6f\taccuracy=%.6f' % ( step, tl/test_data['L'], acc), end='\n')
        print('[Test L]\t%s' % str(la[:16]) )
        print('[Test P]\t%s' % str(pa[:16]) )
        print("-----------------------------------------")
  
      else:
        ## run optimizer
        _, l = session.run([optimizer, cross_entropy], feed_dict=feed_dict)
  
      if (step > 0 and step % param['save_batch'] == 0):
        model_name = "dnn-model-" + start_time + '-' + str(step)
        if not os.path.exists(param['modelpath']):
          os.mkdir(param['modelpath'])
        saver.save(session, os.path.join(param['modelpath'], model_name))
  
  readdata.stop_and_wait_ansyc()

if __name__ == "__main__":
  main()
