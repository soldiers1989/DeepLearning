#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from VedioClassifyInputAnsy import VedioClassifyInputAnsy

import TFBCUtils

param = {
  'ckpt'     : 'D:\\DeepLearning\\model\\',
  'inputpath': 'data/',
  'predset'  : ['cdata', 'cdata'],
  'output'   : 'vedio.pred'
}

param2 = {
  'ckpt'     : '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/classify/model/fmodel/',
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/classify/data/',
  'predset'  : ['test0', 'test1'],
  'output'   : 'vedio.pred'
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
  mid_dim = 256
  output_dim = 28
  output_dim2 = 174

  ##----------------------------input
  with tf.name_scope('input') as scope:
    lda1000 = tf.placeholder(dtype='float', shape=[None, input_dim], name='input_lda1000')
    lda2000 = tf.placeholder(dtype='float', shape=[None, input_dim2], name='input_lda2000')
    lda5000 = tf.placeholder(dtype='float', shape=[None, input_dim5], name='input_lda5000')
    label1 = tf.placeholder(dtype='float', shape=[None, output_dim], name='input_label1')
    label2 = tf.placeholder(dtype='float', shape=[None, output_dim2], name='input_labe2')

  with tf.name_scope('param') as scope:
    keep_prob = tf.placeholder(dtype="float", name='keep_prob')
    global_step = tf.placeholder(dtype=np.int32, name="global_step")
    
  ##----------------------------concat layer
  with tf.name_scope('concat') as scope:
    concat_item = tf.concat([lda1000, lda2000, lda5000], 1)

  ##----------------------------fc layer
  with tf.name_scope('fc') as scope:    
    fc1_w0, fc1_b0 = TFBCUtils.create_w_b(input_dim+input_dim2+input_dim5, mid_dim, w_name="fc1_w0", b_name="fc1_b0")
    fc21_w0, fc21_b0 = TFBCUtils.create_w_b(mid_dim, output_dim, w_name="fc21_w0", b_name="fc21_b0")
    fc22_w0, fc22_b0 = TFBCUtils.create_w_b(mid_dim, output_dim2, w_name="fc22_w0", b_name="fc22_b0")
    
  ##----------------------------fc layer
  with tf.name_scope('fc') as scope:    
    layer1out = tf.nn.relu( tf.matmul(concat_item, fc1_w0) + fc1_b0 )

  ##----------------------------loss layer
  with tf.name_scope('loss') as scope:
    pred1 = tf.nn.softmax( tf.matmul(layer1out, fc21_w0) + fc21_b0 )
    cross_entropy1 = -tf.reduce_sum(label1 * tf.log(pred1))
    
    pred2 = tf.nn.softmax( tf.matmul(layer1out, fc22_w0) + fc22_b0 )
    cross_entropy2 = -tf.reduce_sum(label2 * tf.log(pred2))

  readdata = VedioClassifyInputAnsy(param)
  outfname = param['inputpath'] + '/' + param['output'] + start_time
  
  with tf.Session() as sess, open(outfname, 'w', encoding="utf-8") as outf:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(param['ckpt']))

    predata = readdata.read_preddata_batch()
    outfname = param['inputpath'] + '/' + param['output'] + start_time
    
    while predata['L']>0:  
      ## feed data to tf session
      feed_dict = {
        lda1000: predata['lda1000'],
        lda2000: predata['lda2000'],
        lda5000: predata['lda5000'],
        keep_prob: 1
      }
      
      p1, p2 = sess.run([pred1, pred2], feed_dict=feed_dict)
      for k, r1, r2 in zip(predata['addinfo'], p1, p2):
      	outf.write(','.join([str(x) for x in r1]))
      	outf.write('|')
      	outf.write(','.join([str(x) for x in r2]))
      	outf.write('|' + k + '\n')
      
      predata = readdata.read_preddata_batch()

if __name__ == "__main__":
  main()
