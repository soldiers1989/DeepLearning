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
from SMBAnsyInputV2 import SMBAnsyInput

import TFBCUtils

Py3 = sys.version_info[0] == 3
if not Py3: import codecs

param = {
  'inputpath': 'data/',
  'modelpath': 'model/',
  'dataset': ['smb.data.sample', 'smb.data.sample2'],
  'testset': ['smb.data.sample'],
  'predset': [],

  'shuffle_file': True,
  'batch_size': 16,
  'batch_size_test': 16,
  'test_batch': 100,
  'save_batch': 500,
  'total_batch': 1000,
  'decay_steps': 1000,
  'keep_prob': 0.5,
  'grad_clip': 1.5,
  'lr': 0.0001,
  'loss':'top1',  # 'cross-entropy'   'top1'
  'final_act':'relu',  # 'cross-entropy'   'top1'

  'emb_size': 100,
  'layers': 2,
  'titlemax_size': 20,
  'articlemax_size': 200,

  'vocab': 'data/model2.vec.proc',
  'vocab_size': 1000,
  'kernel_sizes': [2, 3],
  'filters': 2
}

param2 = {
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/smb/data/',
  'modelpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/smb/model/',

  'dataset': ['data1', 'data2', 'data3', 'data4'],
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
  'vocab_size': 283540,
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
    
  def create_rnn(self, x, seqlen, seq_max_len, name):
    with tf.name_scope(name) as scope:
      with tf.variable_scope(name):
        # 输入x的形状： (batch_size, max_seq_len, n_input) 输入seqlen的形状：(batch_size, )
        # 定义一个lstm_cell，隐层的大小为n_hidden（之前的参数）
        # self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.args['emb_size'])

        # 使用tf.nn.dynamic_rnn展开时间维度
        # 此外sequence_length=seqlen也很重要，它告诉TensorFlow每一个序列应该运行多少步
        outputs, states = tf.nn.dynamic_rnn(self.title_lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

        # outputs的形状为(batch_size, max_seq_len, n_hidden)
        # 我们希望的是取出与序列长度相对应的输出。如一个序列长度为10，我们就应该取出第10个输出
        # 但是TensorFlow不支持直接对outputs进行索引，因此我们用下面的方法来做：

        batch_size = tf.shape(outputs)[0]
        # 得到每一个序列真正的index
        # tf.range 创建一个数字序列
        # tf.gather根据索引，从输入张量中依次取元素，构成一个新的张量。
        #                       [0 1 2] * 20          + [ 9, 18, 19] - 1
        index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)  # [ 8 37 58]
        #                   tf.reshape(outputs, [-1, n_hidden])   size: (60, 64)  60=3*20
        outputs = tf.gather(tf.reshape(outputs, [-1, self.args['emb_size']]), index)
        # size: (3, 64) 取了60里的[ 8 37 58]

        return outputs

  def _init_graph(self):
    ##----------------------------input
    with tf.name_scope('input') as scope:
      self.Xbizclass = tf.placeholder(shape=[None, 2], dtype=tf.int32, name='x_input_bizclass')
      self.Xvtitleseg = tf.placeholder(shape=[None, self.args['titlemax_size']], dtype=tf.int32, name='x_input_vtitleseg')
      self.Xvtitlelen = tf.placeholder(shape=[None], dtype=tf.int32, name='x_input_vtitlelen')

      self.Ybizclass = tf.placeholder(shape=[None, 2], dtype=tf.int32, name='y_input_bizclass')
      self.Yvtitleseg = tf.placeholder(shape=[None, self.args['titlemax_size']], dtype=tf.int32, name='y_input_vtitleseg')
      self.Yvtitlelen = tf.placeholder(shape=[None], dtype=tf.int32, name='y_input_vtitlelen')

      self.State = [tf.placeholder(tf.float32, [self.args['batch_size'], self.args['emb_size']], name='rnn_state') for _ in range(self.args['layers'])]

    with tf.name_scope('param') as scope:
      self.keep_prob = tf.placeholder(dtype="float", name='keep_prob')
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    ##----------------------------embedding layer
    with tf.device('/cpu:0'):
      with tf.name_scope('embedding') as scope:
        self.embedding = TFBCUtils.addvocabembedding(self.vocab)

        self.Xbizclassembedding1 = tf.nn.embedding_lookup(self.embedding, self.Xbizclass)
        self.Xbizclassembedding = tf.reshape(self.Xbizclassembedding1, [-1, self.args['emb_size'] * 2])
        self.Xvtitleembedding = tf.nn.embedding_lookup(self.embedding, self.Xvtitleseg)

        self.Ybizclassembedding1 = tf.nn.embedding_lookup(self.embedding, self.Ybizclass)
        self.Ybizclassembedding = tf.reshape(self.Ybizclassembedding1, [-1, self.args['emb_size'] * 2])
        self.Yvtitleembedding = tf.nn.embedding_lookup(self.embedding, self.Yvtitleseg)

    ##----------------------------rnn layer
    with tf.name_scope('rnntitle') as scope:
      self.title_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.args['emb_size'])
      self.Xvtitlernn = self.create_rnn(self.Xvtitleembedding, self.Xvtitlelen, self.args['titlemax_size'], 'vtitlernn')
      self.Yvtitlernn = self.create_rnn(self.Yvtitleembedding, self.Yvtitlelen, self.args['titlemax_size'], 'vtitlernn')

    ##----------------------------concat layer
    with tf.name_scope('concat') as scope:  #  [?,22,100], [300,100]
      self.Xconcat_item = tf.concat([self.Xbizclassembedding, self.Xvtitlernn], 1)
      self.Yconcat_item = tf.concat([self.Ybizclassembedding, self.Yvtitlernn], 1)

    ##----------------------------fc layer
    with tf.name_scope('fc') as scope:
      level1_dim = 3 * self.args['emb_size']
      self.fc1_w0, self.fc1_b0 = TFBCUtils.create_w_b(level1_dim, self.args['emb_size'], w_name="fc1_w0", b_name="fc1_b0")
      self.Xitemembn2 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(self.Xconcat_item, self.fc1_w0) + self.fc1_b0), 1)
      self.Yitemembn2 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(self.Yconcat_item, self.fc1_w0) + self.fc1_b0), 1)

    with tf.name_scope('rnn') as scope:
      self.stacked_cell = rnn_cell.MultiRNNCell([self.get_rnn_cell() for _ in range(self.args['layers'])])
      output, state = self.stacked_cell(self.Xitemembn2, tuple(self.State))
      self.final_state = state
      self.output = tf.nn.l2_normalize(output, 1)

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
        
      self.logits = tf.matmul(self.output, self.Yitemembn2, transpose_b=True)
      self.yhat = self.final_activation(self.logits)
      self.cost = self.loss_function(self.yhat)

      self.learning_rate = tf.train.exponential_decay(0.0002, self.global_step, self.args['decay_steps'], 0.995)

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
          self.Xbizclass: train_data['Xbizclass'], self.Xvtitleseg: train_data['Xvtitleseg'], self.Xvtitlelen: train_data['Xvtitlelen'],
          self.Ybizclass: train_data['Ybizclass'], self.Yvtitleseg: train_data['Yvtitleseg'], self.Yvtitlelen: train_data['Yvtitlelen'],
          self.keep_prob: self.args['keep_prob']
        }
        for jj in range(self.args['layers']):  feed_dict[self.State[jj]] = state[jj]

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

  def infer(self, readdata, outf):
    with tf.Session() as sess:
      self.saver = tf.train.Saver()
      print('Loading model:'+self.args['ckpt'])
      self.saver.restore(sess, self.args['ckpt'])

      predata = readdata.read_preddata_batch(self.args['batch_size'])

      while predata['L'] > 0:
        feed_dict = {
          self.Xbizclass: predata['Xbizclass'], self.Xvtitleseg: predata['Xvtitleseg'], self.Xvtitlelen: predata['Xvtitlelen'],
          self.keep_prob: 1
        }

        vvec = sess.run([self.Xitemembn2], feed_dict=feed_dict)
        for k, r1 in zip(predata['Xlabel'], vvec[0]):
          outf.write( k.replace(' ', '') )
          outf.write( ' ' )
          outf.write( ' '.join([str(x) for x in r1]) )
          outf.write( '\n' )

        predata = readdata.read_preddata_batch(self.args['batch_size'])

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
  parser.add_argument('--predset', nargs='+', default=['pred.sample'],
                      help='Choose a pred dataset.')
  parser.add_argument('--predoutputfile', nargs='?', default='vedio.pred',
                      help='Choose a pred dataset.')
  parser.add_argument('--ckpt', nargs='?', default='D:\\DeepLearning\\model\\rnn-model-20180801142716-500',
                      help='Path to save the model.')

  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  if args.pred:
    param.update(vars(args))
    readdata = SMBAnsyInput(param)
    model = SMBEmb(param, readdata.vocab)

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
    readdata = SMBAnsyInput(param)
    readdata.start_ansyc()
    model = SMBEmb(param, readdata.vocab)
    model.train(readdata)
    readdata.stop_and_wait_ansyc()
