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
from tensorflow.python.ops import rnn_cell
from SMBAnsyInputV2 import SMBAnsyInputV2

import TFBCUtils

Py3 = sys.version_info[0] == 3
if not Py3: import codecs

param = {
  'inputpath': 'data/',
  'modelpath': 'model/',
  'dataset': ['smbtrainsample.v2', 'smbtrainsample.v2'],
  'testset': ['smbtrainsample.v2'],
  'predset': [],

  'shuffle_file': True,
  'batch_size': 16,
  'batch_size_test': 16,
  'test_batch': 100,
  'save_batch': 500,
  'total_batch': 1000,
  'sigma': 0,
  'init_as_normal': False,
  'decay_steps': 1000,
  'keep_prob': 0.5,
  'grad_clip': 1.5,
  'lr': 0.0001,
  'loss': 'top1',  # 'cross-entropy' 'top1' 'bpr'
  'final_act': 'relu',  # 'tanh'

  'item_num': 200001,
  'emb_size': 100, # also used for rnn
  'layers': 2,
  'titlemax_size': 20,
  'articlemax_size': 200,

  'vocab': 'data/model2.vec.proc',
  'vocab_size': 100,
  'kernel_sizes': [2, 3],
  'filters': 2
}

param2 = {
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/smbv2/20180828/train/',
  'modelpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/smbv2/20180828/model/',

  'dataset': ['part-00000','part-00002','part-00004','part-00006','part-00008','part-00010','part-00012','part-00014','part-00016','part-00018',\
              'part-00001','part-00003','part-00005','part-00007','part-00009','part-00011','part-00013','part-00015','part-00017','part-00019'],
  'testset': [],
  'predset': [],

  'batch_size': 128,
  'batch_size_test': 4096,
  'test_batch': 1000,
  'save_batch': 5000,
  'total_batch': 400000,
  'decay_steps': 5000,
  'keep_prob': 0.5,

  'vocab': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/smb/model2.vec.proc',
  'vocab_size': 100,
  'kernel_sizes': [1, 2, 3, 4],
  'filters': 200
}

#param.update(param2)

class SMBEmb():
  def __init__(self, args, vocab):
    self.args = args

    now = datetime.datetime.now()
    self.timestamp = now.strftime("%Y%m%d%H%M%S")
    TFBCUtils.printmap(self.args)

    self.vocab=vocab
    self.output_dim = 28
    self.output_dim2 = 174

    self._init_graph()

  def linear(self, X):
    return X
  def tanh(self, X):
    return tf.nn.tanh(X)
  def softmax(self, X):
    return tf.nn.softmax(X)
  def softmaxth(self, X):
    return tf.nn.softmax(tf.tanh(X))
  def relu(self, X):
    return tf.nn.relu(X)
  def sigmoid(self, X):
    return tf.nn.sigmoid(X)

  def cross_entropy(self, yhat):
    return tf.reduce_mean(-tf.log(tf.diag_part(yhat)+1e-24))
  def bpr(self, yhat):
    yhatT = tf.transpose(yhat)
    return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat)-yhatT)))
  def top1(self, yhat):
    yhatT = tf.transpose(yhat)
    term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat)+yhatT)+tf.nn.sigmoid(yhatT**2), axis=0)
    term2 = tf.nn.sigmoid(tf.diag_part(yhat)**2) / self.args['batch_size']
    return tf.reduce_mean(term1 - term2)

  def get_rnn_cell(self):
    cell = rnn_cell.GRUCell(self.args['emb_size'])
    return rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

  def _init_graph(self):
    ##----------------------------input
    with tf.name_scope('input') as scope:
      self.X = tf.placeholder(tf.int32, [self.args['batch_size']], name='input')
      self.Y = tf.placeholder(tf.int32, [self.args['batch_size']], name='output')
      self.state = [tf.placeholder(tf.float32, [self.args['batch_size'], self.args['emb_size']], name='rnn_state') for _ in range(self.args['layers'])]

    with tf.name_scope('param') as scope:
      self.keep_prob = tf.placeholder(dtype="float", name='keep_prob')
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    ##----------------------------embedding layer
    with tf.device('/cpu:0'):
      with tf.name_scope('embedding') as scope:
        sigma = self.args['sigma'] if self.args['sigma'] != 0 else np.sqrt(6.0 / (self.args['item_num'] + self.args['emb_size']))
        if self.args['init_as_normal']:
          initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
        else:
          initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
        self.embedding = tf.get_variable('xembedding', [self.args['item_num'], self.args['emb_size']], initializer=initializer)
#        self.softmax_W = tf.get_variable('softmax_w', [self.args['item_num'], self.args['emb_size']], initializer=initializer)
#        self.softmax_b = tf.get_variable('softmax_b', [self.args['item_num']], initializer=tf.constant_initializer(0.0))

        self.xemb = tf.nn.embedding_lookup(self.embedding, self.X)
        self.yemb = tf.nn.embedding_lookup(self.embedding, self.Y)
#        self.sampled_W = tf.nn.embedding_lookup(self.softmax_W, self.Y)
#        self.sampled_b = tf.nn.embedding_lookup(self.softmax_b, self.Y)

    with tf.name_scope('rnn') as scope:
      self.stacked_cell = rnn_cell.MultiRNNCell([self.get_rnn_cell() for _ in range(self.args['layers'])])
      self.output, self.final_state = self.stacked_cell(self.xemb, tuple(self.state))

    ##----------------------------loss layer
    with tf.name_scope('loss') as scope:
      if self.args['loss'] == 'cross-entropy':
        if self.args['final_act'] == 'tanh':
          self.final_activation = self.softmaxth
        else:
          self.final_activation = self.softmax
        self.loss_function = self.cross_entropy
      elif self.args['loss'] == 'bpr':
        if self.args['final_act'] == 'linear':
          self.final_activation = self.linear
        elif self.args['final_act'] == 'relu':
          self.final_activation = self.relu
        else:
          self.final_activation = self.tanh
        self.loss_function = self.bpr
      elif self.args['loss'] == 'top1':
        if self.args['final_act'] == 'linear':
          self.final_activation = self.linear
        elif self.args['final_act'] == 'relu':
          self.final_activation = self.relu
        else:
          self.final_activation = self.tanh
        self.loss_function = self.top1
      else:
        raise NotImplementedError


      self.output = tf.nn.l2_normalize(self.output, 1)
      self.yemb = tf.nn.l2_normalize(self.yemb, 1)
      self.logits = tf.matmul(self.output, self.yemb, transpose_b=True)
#      self.logits = tf.matmul(self.output, self.sampled_W, transpose_b=True) + self.sampled_b
      self.yhat = self.final_activation(self.logits)
      self.cost = self.loss_function(self.yhat)

      self.predlogits = tf.matmul(self.output, self.embedding, transpose_b=True)
      self.predyhat = self.final_activation(self.predlogits)

      self.learning_rate = tf.train.exponential_decay(0.00015, self.global_step, self.args['decay_steps'], 0.995)
      #self.learning_rate = tf.train.cosine_decay_restarts(0.0002, self.global_step, self.args['decay_steps'])
      #self.learning_rate =  tf.constant(0.0001)

      self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.tvars = tf.trainable_variables()
      self.gvs = self.optimizer.compute_gradients(self.cost, self.tvars)
      self.capped_gvs = [(tf.clip_by_norm(grad, self.args['grad_clip']), var) for grad, var in self.gvs]
      self.train_op = self.optimizer.apply_gradients(self.capped_gvs, global_step=self.global_step)

  def train(self, readdata):
    with tf.Session(config=config) as session:
      tf.global_variables_initializer().run()
      self.saver = tf.train.Saver()
      step = 0

      state = [np.zeros([self.args['batch_size'], self.args['emb_size']], dtype=np.float32) for _ in range(self.args['layers'])]

      ll=0
      print('Before ENTER loop')
      for step in range(self.args['total_batch']):
        train_data = readdata.read_traindata_batch_ansyc()

        for ii in range(train_data['L']):
          if train_data['restart'][ii]==1:
            for jj in range(self.args['layers']):
              state[jj][ii] = 0

        ## feed data to tf session
        feed_dict = {
          self.X: train_data['Xid'], self.Y: train_data['Yid'],
          self.keep_prob: self.args['keep_prob']
        }
        for jj in range(self.args['layers']):  feed_dict[self.state[jj]] = state[jj]

        if (step > 0 and step % self.args['test_batch'] == 0):
          ## run optimizer
          _, l, state, lr = session.run([self.train_op, self.cost, self.final_state, self.learning_rate ], feed_dict=feed_dict)
          ll+=l
          print('[Train]\tIter:%d\tloss=%.6f\tlr=%.6f\tts=%s'%(step, ll, lr, time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))))
          ll=0
        else:
          _, l, state = session.run([self.train_op, self.cost, self.final_state ], feed_dict=feed_dict)
          ll+=l

        if (step > 0 and step % self.args['save_batch'] == 0):
          model_name = "rnn-model-" + self.timestamp + '-' + str(step)
          if not os.path.exists(self.args['modelpath']):
            os.mkdir(self.args['modelpath'])
          self.saver.save(session, os.path.join(self.args['modelpath'], model_name))

  def printMetrics(self, totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400):
    print('total item: %d'%totalcnt)
    print('avg nrr: %f'%(nrr/totalcnt))
    print('avg recall10: %f'%(1.0*recall10/totalcnt))
    print('avg recall20: %f'%(1.0*recall20/totalcnt))
    print('avg recall40: %f'%(1.0*recall40/totalcnt))
    print('avg recall100: %f'%(1.0*recall100/totalcnt))
    print('avg recall200: %f'%(1.0*recall200/totalcnt))
    print('avg recall400: %f'%(1.0*recall400/totalcnt))

  def computeMetrics(self, itemindex, totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400):
    totalcnt += 1
    nrr+=1.0/itemindex
    if itemindex<=10: recall10+=1
    if itemindex<=20: recall20+=1
    if itemindex<=40: recall40+=1
    if itemindex<=100: recall100+=1
    if itemindex<=200: recall200+=1
    if itemindex<=400: recall400+=1

    if totalcnt%20000==0 and totalcnt>0: self.printMetrics(totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400)

    return totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400

  def infervedio(self, readdata, outf):
    with tf.Session() as sess:
      self.saver = tf.train.Saver()
      print('Loading model:'+self.args['ckpt'])
      self.saver.restore(sess, self.args['ckpt'])

      sw = sess.run([self.softmax_W], feed_dict={ self.keep_prob: 1 })

      predata = readdata.read_predvediodata_batch(self.args['batch_size'])
      while predata['L'] > 0:
        for k, xid in zip(predata['Xlabel'], predata['Xid']):
          if xid>=self.args['item_num']: continue
          r1 = sw[0][xid]
          outf.write( k.replace(' ', '') )
          outf.write( ' ' )
          outf.write( ' '.join([str(x) for x in r1]) )
          outf.write( '\n' )

        predata = readdata.read_predvediodata_batch(self.args['batch_size'])

  def inferuser(self, readdata, outf):
    with tf.Session() as sess:
      self.saver = tf.train.Saver()
      print('Loading model:'+self.args['ckpt'])
      self.saver.restore(sess, self.args['ckpt'])

      state = [np.zeros([self.args['batch_size'], self.args['emb_size']], dtype=np.float32) for _ in range(self.args['layers'])]

      printcnt=0
      totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400=0, 0.0, 0, 0, 0, 0, 0, 0

      predata = readdata.read_preduserdata_batch()
      while predata['L'] > 0:
        for ii in range(predata['L']):
          if predata['restart'][ii]==1:
            #itemidx[ii] = -1
            for jj in range(self.args['layers']):
              state[jj][ii] = 0

        feed_dict = { self.X: predata['Xid'], self.Y: predata['Xid'], self.keep_prob: 1 }
        for jj in range(self.args['layers']):  feed_dict[self.state[jj]] = state[jj]

        state, pred = sess.run([self.final_state, self.predyhat], feed_dict=feed_dict)
        nextpredata = readdata.read_preduserdata_batch()

        for p, r, pr, vid in zip(predata['padding'], nextpredata['restart'], pred, nextpredata['Xid']):
          if p==1 or r==1: continue

          argpred=np.argsort(-pr[1:])+1 #视频下标从1开始
          itemindex = np.argwhere(argpred == vid)[0][0] + 1  #视频下标从1开始

          if printcnt<10:
            printcnt+=1
            print('vid: %s'%str(vid))
            print('argpred: %s'%str(argpred))
            print('itemindex: %s'%str(itemindex))

          totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400 = self.computeMetrics(itemindex, totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400)

        predata = nextpredata

    self.printMetrics(totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400)

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
  parser.add_argument('--predvedioset', nargs='+', default=['smbvedio.v2'], #default=['smbvedio.v2'],  []
                      help='Choose a pred dataset.')
  parser.add_argument('--preduserset', nargs='+', default=['smbuser.v2'],
                      help='Choose a pred dataset.')
  parser.add_argument('--predvediooutput', nargs='?', default='vedio.pred',
                      help='Choose a pred dataset.')
  parser.add_argument('--preduseroutput', nargs='?', default='user.pred',
                      help='Choose a pred dataset.')
  parser.add_argument('--ckpt', nargs='?', default='D:\\DeepLearning\\model\\rnn-model-20180904163344-500',
                      help='Path to save the model.')
  parser.add_argument('--batch_size', type=int, default=16,
                      help='Data batch size')

  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  if args.pred:
    param.update(vars(args))
    readdata = SMBAnsyInputV2(param)
    model = SMBEmb(param, readdata.vocab)

    if len(args.predvedioset)>0:
      outfname = args.inputpath + os.sep + args.predvediooutput

      if Py3:
        with open(outfname, 'w', encoding="utf-8") as outf:
          model.infervedio(readdata, outf)
      else:
        import sys
        reload(sys)
        sys.setdefaultencoding("utf-8")
        with codecs.open(outfname, 'w', encoding='utf-8') as outf:
          print('Using codecs.open')
          model.infervedio(readdata, outf)

    if len(args.preduserset)>0:
      outfname = args.inputpath + os.sep + args.preduseroutput

      if Py3:
        with open(outfname, 'w', encoding="utf-8") as outf:
          model.inferuser(readdata, outf)
      else:
        import sys
        reload(sys)
        sys.setdefaultencoding("utf-8")
        with codecs.open(outfname, 'w', encoding='utf-8') as outf:
          print('Using codecs.open')
          model.inferuser(readdata, outf)


  else:
    readdata = SMBAnsyInputV2(param)
    readdata.start_ansyc()
    model = SMBEmb(param, readdata.vocab)
    model.train(readdata)
    readdata.stop_and_wait_ansyc()
