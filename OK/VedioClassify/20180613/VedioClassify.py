#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import datetime
import os
import sys
import time

import numpy as np
import tensorflow as tf
from VedioClassifyBizuinInputAnsyEmb import VedioClassifyBizuinInputAnsyEmb

import TFBCUtils

Py3 = sys.version_info[0] == 3
if not Py3: import codecs

param = {
  'inputpath': 'data/',
  'modelpath': 'model/',
  'dataset': ['cdatabizuinpic', 'cdatabizuinpic'],
  'testset': ['cdatabizuinpic'],
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
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/classify3/data',
  'modelpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/classify3/model/',

  'dataset': ['train0', 'train1', 'train2', 'train3', 'train4',
              'train5', 'train6', 'train7', 'train8', 'train9'],
  'testset': ['test1', 'test2', 'test3'],
  'predset': [],

  'shuffle_file': True,
  'batch_size': 64,  # 12862
  'batch_size_test': 10240,
  'test_batch': 1000,
  'save_batch': 6000,
  'total_batch': 1000000,
  'decay_steps': 5000,
  'keep_prob': 0.5
}

param.update(param2)

class VedioClassify():
  def __init__(self, args):
    self.args = args

    now = datetime.datetime.now()
    self.timestamp = now.strftime("%Y%m%d%H%M%S")
    print(self.args)

    self.input_dim = 1000
    self.input_dim2 = 2000
    self.input_dim5 = 5000
    self.output_dim = 28
    self.output_dim2 = 174
    self.mid_dim = 256

    self._init_graph()

  def _init_graph(self):
    ##----------------------------input
    with tf.name_scope('input') as scope:
      self.lda1000 = tf.placeholder(dtype='float', shape=[None, self.input_dim], name='input_lda1000')
      self.lda2000 = tf.placeholder(dtype='float', shape=[None, self.input_dim2], name='input_lda2000')
      self.lda5000 = tf.placeholder(dtype='float', shape=[None, self.input_dim5], name='input_lda5000')
      self.bizclass1 = tf.placeholder(dtype='float', shape=[None, self.output_dim], name='input_bizclass1')
      self.bizclass2 = tf.placeholder(dtype='float', shape=[None, self.output_dim2], name='input_bizclass2')
      self.label1 = tf.placeholder(dtype='float', shape=[None, self.output_dim], name='input_label1')
      self.label2 = tf.placeholder(dtype='float', shape=[None, self.output_dim2], name='input_labe2')

    with tf.name_scope('param') as scope:
      self.keep_prob = tf.placeholder(dtype="float", name='keep_prob')
      self.global_step = tf.placeholder(dtype=np.int32, name="global_step")

    ##----------------------------concat layer
    with tf.name_scope('concat') as scope:
      self.concat_item = tf.concat([self.lda1000, self.lda2000, self.lda5000, self.bizclass1, self.bizclass2], 1)

    ##----------------------------fc layer
    with tf.name_scope('fc') as scope:
      level1_dim = self.input_dim + self.input_dim2 + self.input_dim5 + self.output_dim + self.output_dim2
      self.fc1_w0, self.fc1_b0 = TFBCUtils.create_w_b(level1_dim, self.mid_dim, w_name="fc1_w0", b_name="fc1_b0")
      self.fc21_w0, self.fc21_b0 = TFBCUtils.create_w_b(self.mid_dim, self.output_dim, w_name="fc21_w0",
                                                        b_name="fc21_b0")
      self.fc22_w0, self.fc22_b0 = TFBCUtils.create_w_b(self.mid_dim, self.output_dim2, w_name="fc22_w0",
                                                        b_name="fc22_b0")

    ##----------------------------fc layer
    with tf.name_scope('fc') as scope:
      self.layer1out = tf.nn.relu(tf.matmul(self.concat_item, self.fc1_w0) + self.fc1_b0)

    ##----------------------------loss layer
    with tf.name_scope('loss') as scope:
      self.pred1 = tf.nn.softmax(tf.matmul(self.layer1out, self.fc21_w0) + self.fc21_b0)
      self.cross_entropy1 = -tf.reduce_sum(self.label1 * tf.log(self.pred1))

      self.pred2 = tf.nn.softmax(tf.matmul(self.layer1out, self.fc22_w0) + self.fc22_b0)
      self.cross_entropy2 = -tf.reduce_sum(self.label2 * tf.log(self.pred2))

      self.cross_weight1 = tf.constant(0.2, dtype='float')
      self.cross_weight2 = tf.constant(0.8, dtype='float')
      self.cross_entropy = self.cross_weight1 * self.cross_entropy1 + self.cross_weight2 * self.cross_entropy2

      self.learning_rate = tf.train.exponential_decay(0.0002, self.global_step, self.args['decay_steps'], 0.98)
      self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

    ##----------------------------acc compute
    with tf.name_scope('acc') as scope:
      self.pred1_argmax = tf.argmax(self.pred1, 1)
      self.label1_argmax = tf.argmax(self.label1, 1)
      self.correct_prediction1 = tf.equal(self.pred1_argmax, self.label1_argmax)
      self.accuracy1 = tf.reduce_mean(tf.cast(self.correct_prediction1, "float"))

      self.pred2_argmax = tf.argmax(self.pred2, 1)
      self.label2_argmax = tf.argmax(self.label2, 1)
      self.correct_prediction2 = tf.equal(self.pred2_argmax, self.label2_argmax)
      self.accuracy2 = tf.reduce_mean(tf.cast(self.correct_prediction2, "float"))

  def train(self, readdata):
    with tf.Session(config=config) as session:
      tf.global_variables_initializer().run()
      self.saver = tf.train.Saver()
      step = 0

      for step in range(self.args['total_batch']):
        train_data = readdata.read_traindata_batch_ansyc()

        ## feed data to tf session
        feed_dict = {
          self.label1: train_data['label1'],
          self.label2: train_data['label2'],
          self.lda1000: train_data['lda1000'],
          self.lda2000: train_data['lda2000'],
          self.lda5000: train_data['lda5000'],
          self.bizclass1: train_data['bizclass1'],
          self.bizclass2: train_data['bizclass2'],
          self.global_step: step
        }

        if (step > 0 and step % self.args['test_batch'] == 0):
          ## run optimizer
          _, l, cl1, cl2, lr, pa1, la1, pa2, la2, acc1, acc2 = session.run([self.optimizer, \
                                                                            self.cross_entropy, self.cross_entropy1,
                                                                            self.cross_entropy2, self.learning_rate, \
                                                                            self.pred1_argmax, self.label1_argmax,
                                                                            self.pred2_argmax, self.label2_argmax, \
                                                                            self.accuracy1, self.accuracy2], \
                                                                           feed_dict=feed_dict)
          print('[Train]\tIter:%d\tloss=%.6f\tloss1=%.6f\tloss2=%.6f\tlr=%.6f\taccuracy1=%.6f\taccuracy2=%.6f\tts=%s' %
                (step, l / train_data['L'], cl1 / train_data['L'], cl2 / train_data['L'], lr, acc1, acc2, \
                 time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))), end='\n')
          print('[Test L]\t%s' % str(la1[:16]))
          print('[Test P]\t%s' % str(pa1[:16]))
          print('[Test L]\t%s' % str(la2[:16]))
          print('[Test P]\t%s' % str(pa2[:16]))

          test_data = readdata.read_testdata_batch_ansyc()
          feed_dict = {
            self.label1: test_data['label1'],
            self.label2: test_data['label2'],
            self.lda1000: test_data['lda1000'],
            self.lda2000: test_data['lda2000'],
            self.lda5000: test_data['lda5000'],
            self.bizclass1: test_data['bizclass1'],
            self.bizclass2: test_data['bizclass2'],
            self.global_step: step
          }

          ## get acc
          acc1, acc2, tl, pa1, la1, pa2, la2 = session.run([self.accuracy1, self.accuracy2, \
                                                            self.cross_entropy, self.pred1_argmax, self.label1_argmax, \
                                                            self.pred2_argmax, self.label2_argmax], feed_dict=feed_dict)

          print('[Test]\tIter:%d\tloss=%.6f\taccuracy1=%.6f\taccuracy2=%.6f' % (step, tl / test_data['L'], acc1, acc2),
                end='\n')
          print('[Test L]\t%s' % str(la1[:16]))
          print('[Test P]\t%s' % str(pa1[:16]))
          print('[Test L]\t%s' % str(la2[:16]))
          print('[Test P]\t%s' % str(pa2[:16]))
          print("-----------------------------------------")

        else:
          ## run optimizer
          _, l = session.run([self.optimizer, self.cross_entropy], feed_dict=feed_dict)

        if (step > 0 and step % self.args['save_batch'] == 0):
          model_name = "dnn-model-" + self.timestamp + '-' + str(step)
          if not os.path.exists(self.args['modelpath']):
            os.mkdir(self.args['modelpath'])
          self.saver.save(session, os.path.join(self.args['modelpath'], model_name))

  def infer(self, readdata, outf):
    with tf.Session() as sess:
      self.saver = tf.train.Saver()
      print('Loading model:'+self.args['ckpt'])
      #self.saver.restore(sess, tf.train.latest_checkpoint(self.args['ckpt']))
      self.saver.restore(sess, self.args['ckpt'])

      predata = readdata.read_preddata_batch()

      while predata['L'] > 0:
        feed_dict = {
          self.lda1000: predata['lda1000'],
          self.lda2000: predata['lda2000'],
          self.lda5000: predata['lda5000'],
          self.bizclass1: predata['bizclass1'],
          self.bizclass2: predata['bizclass2'],
          self.global_step: 0
        }

        p1, p2 = sess.run([self.pred1, self.pred2], feed_dict=feed_dict)
        for k, r1, r2 in zip(predata['addinfo'], p1, p2):
          maxidx1=np.argmax(r1)
          maxidx2=np.argmax(r2)
          outf.write('%d|%f|%d|%f|'%(maxidx1, r1[maxidx1], maxidx2, r2[maxidx2]))
          outf.write(','.join([str(x) for x in r1]))
          outf.write('|')
          outf.write(','.join([str(x) for x in r2]))
          r1[maxidx1]=0; r2[maxidx2]=0;
          maxidx1=np.argmax(r1)
          maxidx2=np.argmax(r2)
          outf.write('|%d|%f|%d|%f|'%(maxidx1, r1[maxidx1], maxidx2, r2[maxidx2]))
          k=k.replace('_', '|', 1)
          outf.write( k + '\n')

        predata = readdata.read_preddata_batch()

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
  parser = argparse.ArgumentParser(description="Run Vedio Classify NN.")
  parser.add_argument('--pred', type=str2bool, default=False,
                      help='Using pred function.')
  parser.add_argument('--inputpath', nargs='?', default='data/',
                      help='Input data path.')
  parser.add_argument('--predset', nargs='+', default=['cdatabizuinpic'],
                      help='Choose a pred dataset.')
  parser.add_argument('--predoutputfile', nargs='?', default='vedio.pred',
                      help='Choose a pred dataset.')
  parser.add_argument('--ckpt', nargs='?', default='D:\\DeepLearning\\model\\dnn-model-20180613143500-500',
                      help='Path to save the model.')

  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  if args.pred:
    param.update(vars(args))
    model = VedioClassify(param)

    readdata = VedioClassifyBizuinInputAnsyEmb(param)
    outfname = args.inputpath + os.sep + args.predoutputfile

    if Py3:
      with open(outfname, 'w', encoding="utf-8") as outf:
        model.infer(readdata, outf)
    else:
      import sys
      reload(sys)
      sys.setdefaultencoding("utf-8")
      with codecs.open(outfname, 'w', encoding='utf-8') as outf:
        print('Using codecs.open')
        model.infer(readdata, outf)

  else:
    readdata = VedioClassifyBizuinInputAnsyEmb(param)
    readdata.start_ansyc()
    model = VedioClassify(param)
    model.train(readdata)
    readdata.stop_and_wait_ansyc()
