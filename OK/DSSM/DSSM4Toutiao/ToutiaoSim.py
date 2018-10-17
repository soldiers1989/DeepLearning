#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import datetime
import os
import time
import tensorflow as tf

import TFBCUtils
from TFBCUtils import Vocab
from ToutiaoSimInput import ToutiaoSimInput

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
  
class ToutiaoSim():
  def __init__(self, args, vocab):
    self.args = args

    now = datetime.datetime.now()
    self.timestamp = now.strftime("%Y%m%d%H%M%S")
    TFBCUtils.printmap(self.args)

    self.vocab=vocab

    print('Graph initing')
    self._init_graph()
    print('Graph inited')
    
  def create_seg_len(self, max_size, name, scopename='seglen' ):
    with tf.name_scope(scopename) as scope:
      seg = tf.placeholder(shape=[None, max_size], dtype=tf.int32, name='%sseg'%name)
      length = tf.placeholder(shape=[None], dtype=tf.int32, name='%slen'%name)
      return seg, length 
  
  def conv_content(self, x, cnn_w, cnn_b, clen, ksize):    
    with tf.name_scope('conv') as scope:
      c = tf.add(tf.nn.conv2d(x, cnn_w, [1, 1, 1, 1], padding='VALID'), cnn_b)
      c = tf.nn.relu(c)
      m = tf.nn.max_pool(c, [1, clen - ksize + 1, 1, 1], [1, 1, 1, 1], padding='VALID')
      m = tf.squeeze(m, [1, 2])
      return m  

  def _init_graph(self):
    ##----------------------------input
    with tf.name_scope('input') as scope:
      self.titleseg, self.titlelen = self.create_seg_len(self.args['titlemax_size'], 'title') 
      self.contentseg, self.contentlen = self.create_seg_len(self.args['contentmax_size'], 'content') 
      
      self.postitleseg, self.postitlelen = self.create_seg_len(self.args['titlemax_size'], 'postitle') 
      self.poscontentseg, self.poscontentlen = self.create_seg_len(self.args['contentmax_size'], 'poscontent')
      
      self.negtitleseg, self.negtitlelen = self.create_seg_len(self.args['titlemax_size'], 'negtitle') 
      self.negcontentseg, self.negcontentlen = self.create_seg_len(self.args['contentmax_size'], 'negcontent')

    with tf.name_scope('param') as scope:
      self.keep_prob = tf.placeholder(dtype="float", name='keep_prob')
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    ##----------------------------embedding layer
    with tf.device('/cpu:0'):
      with tf.name_scope('embedding') as scope:
        self.embedding = TFBCUtils.addvocabembedding_with_zero(self.vocab)        
        
        self.titleemb = tf.nn.embedding_lookup(self.embedding, self.titleseg)
        self.titleemb3d = tf.expand_dims(self.titleemb, -1)        
        self.contentemb = tf.nn.embedding_lookup(self.embedding, self.contentseg)
        self.contentemb3d = tf.expand_dims(self.contentemb, -1)        
        self.postitleemb = tf.nn.embedding_lookup(self.embedding, self.postitleseg)
        self.postitleemb3d = tf.expand_dims(self.postitleemb, -1)        
        self.poscontentemb = tf.nn.embedding_lookup(self.embedding, self.poscontentseg)
        self.poscontentemb3d = tf.expand_dims(self.poscontentemb, -1)        
        self.negtitleemb = tf.nn.embedding_lookup(self.embedding, self.negtitleseg)
        self.negtitleemb3d = tf.expand_dims(self.negtitleemb, -1)
        self.negcontentemb = tf.nn.embedding_lookup(self.embedding, self.negcontentseg)
        self.negcontentemb3d = tf.expand_dims(self.negcontentemb, -1)
               
    ##----------------------------conv layer
    with tf.name_scope('conv') as scope:
      self.convparam = []
      self.convresult = []
      self.convposresult = []
      self.convnegresult = []
      
      for ksizes in self.args['kernel_sizes']:
        cnn_w0 = tf.Variable(tf.random_uniform([ksizes, self.args['emb_size'], 1, self.args['filters']], -0.2, 0.2),
                             dtype='float32', name="cnn_%d_w0" % ksizes)
        cnn_b0 = tf.Variable(tf.constant(0.00001, shape=[self.args['filters']]),
                             dtype='float32', name="cnn_%d_b0" % ksizes)
        self.convparam.append((cnn_w0, cnn_b0))

        titlemax=self.conv_content(self.titleemb3d, cnn_w0, cnn_b0, self.args['titlemax_size'], ksizes)
        contentmax=self.conv_content(self.contentemb3d, cnn_w0, cnn_b0, self.args['contentmax_size'], ksizes)
        self.convresult.append(tf.concat([titlemax, contentmax], 1))
        
        postitlemax=self.conv_content(self.postitleemb3d, cnn_w0, cnn_b0, self.args['titlemax_size'], ksizes)
        poscontentmax=self.conv_content(self.poscontentemb3d, cnn_w0, cnn_b0, self.args['contentmax_size'], ksizes)
        self.convposresult.append(tf.concat([postitlemax, poscontentmax], 1))
        
        negtitlemax=self.conv_content(self.negtitleemb3d, cnn_w0, cnn_b0, self.args['titlemax_size'], ksizes)
        negcontentmax=self.conv_content(self.negcontentemb3d, cnn_w0, cnn_b0, self.args['contentmax_size'], ksizes)
        self.convnegresult.append(tf.concat([negtitlemax, negcontentmax], 1))
        
    ##----------------------------concat layer
    with tf.name_scope('concat') as scope:
      self.concatitem = tf.concat( self.convresult, 1)
      self.concatpositem = tf.concat( self.convposresult, 1)
      self.concatnegitem = tf.concat( self.convnegresult, 1)

    ##----------------------------fc layer
    with tf.name_scope('fc') as scope:
      level1_dim = 2 * self.args['filters'] * len(self.args['kernel_sizes'])
      self.fc1_w0, self.fc1_b0 = TFBCUtils.create_w_b(level1_dim, self.args['emb_size'], w_name="fc1_w0", b_name="fc1_b0")
      self.itemmembn2 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(self.concatitem, self.fc1_w0) + self.fc1_b0), 1)
      self.positembn2 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(self.concatpositem, self.fc1_w0) + self.fc1_b0), 1)
      self.negitembn2 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(self.concatnegitem, self.fc1_w0) + self.fc1_b0), 1)

    ##----------------------------loss layer
    with tf.name_scope('loss') as scope:
      self.cost = TFBCUtils.compute_triplet_loss(self.itemmembn2, self.positembn2, self.negitembn2, self.args['margin'])

      self.learning_rate = tf.train.exponential_decay(0.00015, self.global_step, self.args['decay_steps'], 0.995)
      #self.learning_rate = tf.train.cosine_decay_restarts(0.0002, self.global_step, self.args['decay_steps'])
      #self.learning_rate =  tf.constant(0.0001)

      self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.tvars = tf.trainable_variables()
      self.gvs = self.optimizer.compute_gradients(self.cost, self.tvars)
      self.capped_gvs = [(tf.clip_by_norm(grad, self.args['grad_clip']), var) for grad, var in self.gvs]
      self.train_op = self.optimizer.apply_gradients(self.capped_gvs, global_step=self.global_step)

  def savemodel(self, step, session):
    model_name = "org_smb_model" + self.timestamp + '-' + str(step)
    if not os.path.exists(self.args['modelpath']):
      os.mkdir(self.args['modelpath'])
    self.saver.save(session, os.path.join(self.args['modelpath'], model_name))

  def train(self, readdata):
    with tf.Session(config=config) as session:
      print('Before tf.global_variables_initializer()')
      tf.global_variables_initializer().run()
      self.saver = tf.train.Saver()
      ll=0
      print('Before readdata.read_traindata_batch_ansyc')
      train_data = readdata.read_traindata_batch_ansyc()
      print('Before ENTER loop')
      for step in range(self.args['total_batch']):
        train_data = readdata.read_traindata_batch_ansyc()

        ## feed data to tf session
        feed_dict = { self.keep_prob: self.args['keep_prob'],
          #titlelen CNN不需要
          self.titleseg: train_data['titleseg'], self.titlelen: train_data['titlelen'], 
          self.contentseg: train_data['contentseg'], self.contentlen: train_data['contentlen'],      
          self.postitleseg: train_data['postitleseg'], self.postitlelen: train_data['postitlelen'],
          self.poscontentseg: train_data['poscontentseg'], self.poscontentlen: train_data['poscontentlen'],      
          self.negtitleseg: train_data['negtitleseg'], self.negtitlelen: train_data['negtitlelen'],
          self.negcontentseg: train_data['negcontentseg'], self.negcontentlen: train_data['negcontentlen'],          
        }

        if step > 0 and step % self.args['test_batch'] == 0:
          ## run optimizer
          _, l, lr = session.run([self.train_op, self.cost, self.learning_rate ], feed_dict=feed_dict)
          ll+=l
          print('[Train]\tIter:%d\tloss=%.6f\tlr=%.6f\tts=%s'%(step, ll, lr, time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))))
          ll=0
        else:
          _, l = session.run([self.train_op, self.cost ], feed_dict=feed_dict)
          ll+=l

        if (step > 0 and step % self.args['save_batch'] == 0):
          model_name = "toutiao-model-" + self.timestamp + '-' + str(step)
          if not os.path.exists(self.args['modelpath']):
            os.mkdir(self.args['modelpath'])
          self.saver.save(session, os.path.join(self.args['modelpath'], model_name))

  def infermsg(self, readdata, outf):
    with tf.Session() as sess:
      self.saver = tf.train.Saver()
      print('Loading model:'+self.args['ckpt'])
      self.saver.restore(sess, self.args['ckpt'])

      predata = readdata.read_preddata_batch(self.args['batch_size'])
      while predata['L'] > 0:
        feed_dict = {
          self.postitleseg: predata['titleseg'], self.postitlelen: predata['titlelen'],
          self.poscontentseg: predata['contentseg'], self.poscontentlen: predata['contentlen'],
          self.keep_prob: 1
        }
        
        vvec = sess.run([self.positembn2], feed_dict=feed_dict)
        for k, r1 in zip(predata['addinfo'], vvec[0]):
          outf.write( k.replace(' ', '').replace('"', '') )
          outf.write( ' ' )
          outf.write( ' '.join([str(x) for x in r1]) )
          outf.write( '\n' )

        predata = readdata.read_preddata_batch(self.args['batch_size'])

