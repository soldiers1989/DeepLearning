#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import datetime
import math
import os
import sys

import numpy as np
import tensorflow as tf
from VedioTagNCEInputAnsy import VedioTagNCEInputAnsy

import TFBCUtils

Py3 = sys.version_info[0] == 3
if not Py3: import codecs

param = {
  'inputpath': 'data/',
  'modelpath': 'model/',
  'dataset': ['tagdata', 'tagdata'],
  'testset': ['tagdata'],
  'predset': [],

  'shuffle_file': True,
  'batch_size': 16,
  'batch_size_test': 16,
  'test_batch': 100,
  'save_batch': 500,
  'total_batch': 1000,
  'learning_rate': 0.001,
  'decay_steps': 1000,
  'keep_prob': 0.7,

  'emb_size': 100,
  'titlemax_size': 20,
  'articlemax_size': 200,

  'vocab': 'data/model2.vec.proc',
  'vocab_size': 1000,
  'kernel_sizes': [2, 3],
  'filters': 2,

  'tag_size': 8000,
  'num_sampled': 2
}

param2 = {
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/tag1/data/',
  'modelpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/tag1/model/',

  'dataset': ['train0', 'train1', 'train2', 'train3', 'train4',
              'train5', 'train6', 'train7', 'train8', 'train9',
              'train10', 'train11', 'train12', 'train13', 'train14'],
  'testset': ['test1', 'test2', 'test3', 'test4', 'test5', 'test6'],
  'predset': [],

  'batch_size': 128,
  'batch_size_test': 1024,
  'test_batch': 1000,
  'save_batch': 5000,
  'total_batch': 1000000,
  'decay_steps': 5000,
  'keep_prob': 0.5,

  'vocab': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/mpvedio/classfy4/w2v/model2.vec.proc',
  'vocab_size': 0,
  'kernel_sizes': [1, 2, 3, 4],
  'filters': 200,

  'num_sampled': 100
}

param.update(param2)

class VedioClassify():
  def __init__(self, args, vocab):
    self.args = args

    now = datetime.datetime.now()
    self.timestamp = now.strftime("%Y%m%d%H%M%S")
    print(self.args)

    self.vocab = vocab

    self.input_dim = 1000
    self.input_dim2 = 2000
    self.input_dim5 = 5000
    self.mid_dim = 256
    self.output_dim = self.args['emb_size']

    self._init_graph()

  def _init_graph(self):
    ##----------------------------input
    with tf.name_scope('input') as scope:
      self.lda1000 = tf.placeholder(dtype='float', shape=[None, self.input_dim], name='input_lda1000')
      self.lda2000 = tf.placeholder(dtype='float', shape=[None, self.input_dim2], name='input_lda2000')
      self.lda5000 = tf.placeholder(dtype='float', shape=[None, self.input_dim5], name='input_lda5000')

      self.titleseg = tf.placeholder(shape=[None, self.args['titlemax_size']], dtype=tf.int32, name='input_titleseg')
      self.vtitleseg = tf.placeholder(shape=[None, self.args['titlemax_size']], dtype=tf.int32, name='input_vtitleseg')
      self.contentseg = tf.placeholder(shape=[None, self.args['articlemax_size']], dtype=tf.int32,
                                       name='input_contentseg')

      self.bizclass1 = tf.placeholder(dtype='float', shape=[None, 28], name='input_bizclass1')
      self.bizclass2 = tf.placeholder(dtype='float', shape=[None, 174], name='input_bizclass2')

      self.label = tf.placeholder(dtype=tf.int64, shape=[None, 1], name='input_label')

    with tf.name_scope('param') as scope:
      self.keep_prob = tf.placeholder(dtype="float", name='keep_prob')
      self.global_step = tf.placeholder(dtype=np.int32, name="global_step")

    ##----------------------------embedding layer
    with tf.device('/cpu:0'), tf.name_scope('embedding') as scope:
      self.embedding = TFBCUtils.addvocabembedding(self.vocab)

      self.titleembedding = tf.expand_dims(tf.nn.embedding_lookup(self.embedding, self.titleseg), -1)
      self.vtitleembedding = tf.expand_dims(tf.nn.embedding_lookup(self.embedding, self.vtitleseg), -1)
      self.contentbedding = tf.expand_dims(tf.nn.embedding_lookup(self.embedding, self.contentseg), -1)

    ##----------------------------conv layer
    with tf.name_scope('conv') as scope:
      self.convparam = []
      self.convresult = []
      for kernel_sizes in self.args['kernel_sizes']:
        cnn_w0 = tf.Variable(
          tf.random_uniform([kernel_sizes, self.args['emb_size'], 1, self.args['filters']], -0.2, 0.2),
          dtype='float32', name="cnn_%d_w0" % kernel_sizes)
        cnn_b0 = tf.Variable(tf.constant(0.00001, shape=[self.args['filters']]),
                             name="cnn_%d_b0" % kernel_sizes)
        self.convparam.append((cnn_w0, cnn_b0))

        titlecnn = tf.add(tf.nn.conv2d(self.titleembedding, cnn_w0, [1, 1, 1, 1], padding='VALID'), cnn_b0)
        titlecnn = tf.nn.relu(titlecnn)
        titlemax = tf.nn.max_pool(titlecnn, [1, self.args['titlemax_size'] - kernel_sizes + 1, 1, 1], [1, 1, 1, 1],
                                  padding='VALID')
        titlemax = tf.squeeze(titlemax, [1, 2])

        vtitlecnn = tf.add(tf.nn.conv2d(self.vtitleembedding, cnn_w0, [1, 1, 1, 1], padding='VALID'), cnn_b0)
        vtitlecnn = tf.nn.relu(vtitlecnn)
        vtitlemax = tf.nn.max_pool(vtitlecnn, [1, self.args['titlemax_size'] - kernel_sizes + 1, 1, 1], [1, 1, 1, 1],
                                   padding='VALID')
        titlemax = tf.squeeze(vtitlemax, [1, 2])

        contentcnn = tf.add(tf.nn.conv2d(self.contentbedding, cnn_w0, [1, 1, 1, 1], padding='VALID'), cnn_b0)
        contentcnn = tf.nn.relu(contentcnn)
        contentmax = tf.nn.max_pool(contentcnn, [1, self.args['articlemax_size'] - kernel_sizes + 1, 1, 1],
                                    [1, 1, 1, 1], padding='VALID')
        contentmax = tf.squeeze(contentmax, [1, 2])

        mergered = tf.concat([titlemax, titlemax, contentmax], 1)
        self.convresult.append(mergered)

    ##----------------------------concat layer
    with tf.name_scope('concat') as scope:
      self.concat_item = tf.concat(
        [self.lda1000, self.lda2000, self.lda5000, self.bizclass1, self.bizclass2] + self.convresult, 1)

    ##----------------------------fc layer
    with tf.name_scope('fc') as scope:
      level1_dim = self.input_dim + self.input_dim2 + self.input_dim5 + 28 + 174
      level1_dim += 3 * self.args['filters'] * len(self.args['kernel_sizes'])
      self.fc1_w0, self.fc1_b0 = TFBCUtils.create_w_b(level1_dim, self.mid_dim, w_name="fc1_w0", b_name="fc1_b0")
      self.fc2_w0, self.fc2_b0 = TFBCUtils.create_w_b(self.mid_dim, self.output_dim, w_name="fc2_w0", b_name="fc2_b0")

      self.layer1out = tf.nn.relu(tf.matmul(self.concat_item, self.fc1_w0) + self.fc1_b0)
      self.layer1out = tf.nn.dropout(self.layer1out, self.keep_prob)
      self.layer2out = tf.nn.relu(tf.matmul(self.layer1out, self.fc2_w0) + self.fc2_b0)
      self.layer2out = tf.nn.dropout(self.layer2out, self.keep_prob)

    ##----------------------------nce data
    with tf.device('/cpu:0'), tf.name_scope('nec') as scope:
      # Construct the variables for the NCE loss
      self.nce_weights = tf.Variable(
        tf.truncated_normal([self.args['tag_size'], self.args['emb_size']],
                            stddev=1.0 / math.sqrt(self.args['emb_size'])))
      self.nce_biases = tf.Variable(tf.zeros([self.args['tag_size']]))

      ##----------------------------loss layer
    with tf.name_scope('loss') as scope:
      self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weights,
                                                biases=self.nce_biases,
                                                labels=self.label,
                                                inputs=self.layer2out,
                                                num_sampled=self.args['num_sampled'],
                                                num_classes=self.args['tag_size']))

      self.learning_rate = tf.train.exponential_decay(self.args['learning_rate'], self.global_step, self.args['decay_steps'], 0.98)
      self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    ##----------------------------acc compute
    with tf.name_scope('acc') as scope:
      self.output = tf.matmul(self.layer2out, tf.transpose(self.nce_weights)) + self.nce_biases

  def train(self, readdata):
    with tf.Session(config=config) as session:
      tf.global_variables_initializer().run()
      self.saver = tf.train.Saver()
      step = 0
  
      for step in range(self.args['total_batch']):
        train_data = readdata.read_traindata_batch_ansyc()
  
        ## feed data to tf session
        feed_dict = {
          self.label: train_data['label'],
          self.lda1000: train_data['lda1000'], self.lda2000: train_data['lda2000'], self.lda5000: train_data['lda5000'],
          self.bizclass1: train_data['bizclass1'], self.bizclass2: train_data['bizclass2'],
          self.titleseg: train_data['titleseg'], self.vtitleseg: train_data['vtitleseg'],
          self.contentseg: train_data['contentseg'],
          self.keep_prob: self.args['keep_prob'], self.global_step: step
        }
  
        if (step > 0 and step % self.args['test_batch'] == 0):
          ## run optimizer
          _, lr, ls, out = session.run([self.optimizer, self.learning_rate, self.loss, self.output],
                                       feed_dict=feed_dict)
          acc1, acc3, pred = TFBCUtils.top3acc(out, [x[0] for x in train_data['label']])
          print('[Train]\tIter:%d\tloss=%.6f\tlr=%.6f\tacc=%.6f\tacc3=%.6f' % (step, ls/train_data['L'], lr, acc1, acc3), end='\n')
          print('[Train P]\t%s' % str(pred[:16]))
          print('[Train L]\t%s' % str([x[0] for x in train_data['label'][:16]]))
          print('[Train L]\t%s' % str(out[:5]))
        
          test_data = readdata.read_testdata_batch_ansyc()
          feed_dict = {
            self.label: test_data['label'],
            self.lda1000: test_data['lda1000'], self.lda2000: test_data['lda2000'], self.lda5000: test_data['lda5000'],
            self.bizclass1: test_data['bizclass1'], self.bizclass2: test_data['bizclass2'],
            self.titleseg: test_data['titleseg'], self.vtitleseg: test_data['vtitleseg'],
            self.contentseg: test_data['contentseg'],
            self.keep_prob: 1.0, self.global_step: step
          }
  
          ## get acc
          ls, out = session.run([self.loss, self.output], feed_dict=feed_dict)
          acc1, acc3, pred = TFBCUtils.top3acc(out, [x[0] for x in test_data['label']])           
          print('[Test]\tIter:%d\tloss=%.6f\tacc=%.6f\tacc3=%.6f' % (step, ls / test_data['L'], acc1, acc3),
                end='\n')
          print('[Test P]\t%s' % str(pred[:16]))
          print('[Test L]\t%s' % str([x[0] for x in test_data['label'][:16]]))
          print('[Test L]\t%s' % str(out[:5]))
          print("-----------------------------------------")
  
        else:
          ## run optimizer
          _, _ = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
  
        if (step > 0 and step % self.args['save_batch'] == 0):
          model_name = "dnn-model-" + self.timestamp + '-' + str(step)
          if not os.path.exists(self.args['modelpath']):
            os.mkdir(self.args['modelpath'])
          self.saver.save(session, os.path.join(self.args['modelpath'], model_name))
  
  
  def infer(self, readdata, outf):
    with tf.Session() as sess:
      self.saver = tf.train.Saver()
      print('Loading model:' + self.args['ckpt'])
      # self.saver.restore(sess, tf.train.latest_checkpoint(self.args['ckpt']))
      self.saver.restore(sess, self.args['ckpt'])
  
      predata = readdata.read_preddata_batch()
  
      while predata['L'] > 0:
        feed_dict = {
          self.lda1000: predata['lda1000'], self.lda2000: predata['lda2000'], self.lda5000: predata['lda5000'],
          self.bizclass1: predata['bizclass1'], self.bizclass2: predata['bizclass2'],
          self.titleseg: predata['titleseg'], self.vtitleseg: predata['vtitleseg'],
          self.contentseg: predata['contentseg'],
          self.keep_prob: 1,
          self.global_step: 0
        }
  
        p1, p2 = sess.run([self.pred1, self.pred2], feed_dict=feed_dict)
        for k, r1, r2 in zip(predata['addinfo'], p1, p2):
          maxidx1 = np.argmax(r1)
          maxidx2 = np.argmax(r2)
          outf.write('%d|%f|%d|%f|' % (maxidx1, r1[maxidx1], maxidx2, r2[maxidx2]))
          outf.write(','.join([str(x) for x in r1]))
          outf.write('|')
          outf.write(','.join([str(x) for x in r2]))
          r1[maxidx1] = 0;
          r2[maxidx2] = 0;
          maxidx1 = np.argmax(r1)
          maxidx2 = np.argmax(r2)
          outf.write('|%d|%f|%d|%f|' % (maxidx1, r1[maxidx1], maxidx2, r2[maxidx2]))
          k = k.replace('_', '|', 1)
          outf.write(k + '\n')
  
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
    readdata = VedioTagNCEInputAnsy(param)
    model = VedioClassify(param, readdata.vocab)

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
    readdata = VedioTagNCEInputAnsy(param)
    readdata.start_ansyc()
    model = VedioClassify(param, readdata.vocab)
    model.train(readdata)
    readdata.stop_and_wait_ansyc()
