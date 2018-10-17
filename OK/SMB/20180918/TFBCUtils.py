#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score
import math
import codecs
import sys
Py3 = sys.version_info[0] == 3

def printmap(amap):
  for key, values in amap.items():
    print('%s --> %s'%(str(key),str(values)))

def onehot (x, n):
  targets = np.array(x).reshape(-1)
  return np.eye(n)[targets]

def tanh_y (x, w, b, keep_prob=1.0):
  y = tf.matmul(x, w) + b
  y = tf.nn.dropout(y, keep_prob)
  return tf.nn.tanh(y)
  
def relu_y (x, w, b, keep_prob=1.0):
  y = tf.matmul(x, w) + b
  y = tf.nn.dropout(y, keep_prob)
  return tf.nn.relu(y)
  
def softsign_y (x, w, b, keep_prob=1.0):
  y = tf.matmul(x, w) + b
  y = tf.nn.dropout(y, keep_prob)
  return tf.nn.softsign(y)
  
def bpr(self, yhat):
  yhatT = tf.transpose(yhat)
  return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat)-yhatT)))

def top1(self, yhat):
  yhatT = tf.transpose(yhat)
  term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat)+yhatT)+tf.nn.sigmoid(yhatT**2), axis=0)
  term2 = tf.nn.sigmoid(tf.diag_part(yhat)**2) / self.args['batch_size']
  return tf.reduce_mean(term1 - term2)

def create_w_b(in_size, out_size, w_name="w", b_name="b", fc_type=tf.float32):
  if (in_size < 1 or out_size < 1):
    return "layer size ERROR"

  left  = - math.sqrt(6) / math.sqrt(in_size + out_size)
  right =   math.sqrt(6) / math.sqrt(in_size + out_size)

  w = tf.Variable(tf.random_uniform([in_size, out_size], left, right), dtype=fc_type, name=w_name)
  b = tf.Variable(tf.zeros([out_size]), dtype=fc_type, name=b_name)
  return w,b

def add_fc_layer(x, in_size, out_size, w_name="w", b_name="b", fc_type=tf.float32, keep_prob=1.0):
  if (in_size < 1 or out_size < 1):
    return "layer size ERROR"

  left  = - math.sqrt(6) / math.sqrt(in_size + out_size)
  right =   math.sqrt(6) / math.sqrt(in_size + out_size)

  w = tf.Variable(tf.random_uniform([in_size, out_size], left, right), dtype=fc_type, name=w_name)
  b = tf.Variable(tf.zeros([out_size]), dtype=fc_type, name=b_name)

  y = tf.matmul(x, w) + b
  y = tf.nn.dropout(y, keep_prob)
  return tf.nn.relu(y)

def compute_triplet_loss(title_feature, pos_feature, neg_feature, margin="0.1"):
  with tf.name_scope("triplet_loss"):
    sim_pos = tf.reduce_sum(tf.multiply(title_feature, pos_feature), 1)
    sim_neg = tf.reduce_sum(tf.multiply(title_feature, neg_feature), 1)
    loss = tf.maximum(0., sim_neg - sim_pos + margin)
    sim_loss = 1 - sim_pos
    return tf.reduce_mean(loss)

def compute_contrastive_loss(model1, model2, y, margin):
  with tf.name_scope("contrastive_loss"):
    d = tf.sqrt( tf.reduce_sum( tf.square( tf.subtract(model1, model2) ), 1, keep_dims=True) )
    tmp= y * tf.square(d)
    tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
    return tf.reduce_mean(tmp + tmp2) /2

def get_cnn_feature(f1, f2, f3, b1, b2, b3, title, sentence_len, kernel_sizes):
  #Convolutions
  C1_title = tf.add(tf.nn.conv2d(title, f1, [1,1,1,1], padding='VALID'), b1)
  C2_title = tf.add(tf.nn.conv2d(title, f2, [1,1,1,1], padding='VALID'), b2)
  C3_title = tf.add(tf.nn.conv2d(title, f3, [1,1,1,1], padding='VALID'), b3)
  C1_title = tf.nn.relu(C1_title)
  C2_title = tf.nn.relu(C2_title)
  C3_title = tf.nn.relu(C3_title)

  ##title Max pooling
  maxC1_title = tf.nn.max_pool(C1_title, [1, sentence_len-kernel_sizes[0]+1,1,1] , [1,1,1,1], padding='VALID')
  maxC1_title = tf.squeeze(maxC1_title, [1, 2])
  maxC2_title = tf.nn.max_pool(C2_title, [1, sentence_len-kernel_sizes[1]+1,1,1] , [1,1,1,1], padding='VALID')
  maxC2_title = tf.squeeze(maxC2_title, [1,2])
  maxC3_title = tf.nn.max_pool(C3_title, [1, sentence_len-kernel_sizes[2]+1,1,1] , [1,1,1,1], padding='VALID')
  maxC3_title = tf.squeeze(maxC3_title, [1,2])

  #title Concatenating pooled features
  mergered = tf.concat([maxC1_title, maxC2_title, maxC3_title], 1)
  return mergered

def create_cnn_param(kernel_sizes, edim, n_filters):
  F1  = tf.Variable(tf.random_uniform([kernel_sizes[0], edim, 1, n_filters], -0.2, 0.2),dtype='float32')
  F2  = tf.Variable(tf.random_uniform([kernel_sizes[1], edim, 1, n_filters], -0.2, 0.2),dtype='float32')
  F3  = tf.Variable(tf.random_uniform([kernel_sizes[2], edim, 1, n_filters], -0.2, 0.2),dtype='float32')
  FB1 = tf.Variable(tf.constant(0.00001, shape=[n_filters]))
  FB2 = tf.Variable(tf.constant(0.00001, shape=[n_filters]))
  FB3 = tf.Variable(tf.constant(0.00001, shape=[n_filters]))
  return F1, F2, F3, FB1, FB2, FB3

def get_gru_cell(self, size, keep_prob):
  cell = rnn_cell.GRUCell(size)
  return rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

def create_rnn(self, cell, emb_size, x, seqlen, seq_max_len, name):
  with tf.name_scope(name) as scope:
    with tf.variable_scope(name):
      # 输入x的形状： (batch_size, max_seq_len, n_input) 输入seqlen的形状：(batch_size, )
      # 定义一个lstm_cell，隐层的大小为n_hidden（之前的参数）
      # self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.args['emb_size'])

      # 使用tf.nn.dynamic_rnn展开时间维度
      # 此外sequence_length=seqlen也很重要，它告诉TensorFlow每一个序列应该运行多少步
      outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=seqlen)

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
      outputs = tf.gather(tf.reshape(outputs, [-1, emb_size]), index)
      # size: (3, 64) 取了60里的[ 8 37 58]

      return outputs, states    
    
    
class Vocab(object):
  # Three files are needed in the path
  def __init__(self, filename, emb_size, vocabulary_size=0):
    self.filename=filename
    self.emb_size=emb_size
    self.zero_ebm=np.zeros(emb_size)
    self.vocabulary_size=vocabulary_size
    self.id2stringmap={}
    self.id2vectormap={}
    self.string2idmap={}

    self.loadfile()
    print('Vocab loaded, filename %s'%self.filename)
    
  def loadfile(self):
    def loadfileinner(inf):
      max_wid=0
      for line in inf.readlines():
        items=line.strip().replace('\t',' ').split(' ')
        if len(items)!=self.emb_size+2:
          print('Error in Vocab.loadfile %d %s' % (len(items), line.strip()))
          continue
        
        wid=int(items[0])
        wd=items[1]
        data = [float(item) for item in items[2:]]
        if len(data)!=self.emb_size:
          print('Error in Vocab.loadfile len(data)!=emb_size %d %s' % (len(items), line.strip()))
          continue
          
        if self.vocabulary_size>0 and wid<=self.vocabulary_size:          
          self.id2stringmap[wid]=wd
          self.id2vectormap[wid]=np.array(data) 
          self.string2idmap[wd]=wid
          
        if self.vocabulary_size==0:          
          self.id2stringmap[wid]=wd
          self.id2vectormap[wid]=np.array(data) 
          self.string2idmap[wd]=wid
          if wid>max_wid: max_wid=wid
      
      if self.vocabulary_size==0: self.vocabulary_size=max_wid         
    
    if Py3:
      with open(self.filename, 'r', encoding="utf-8") as inf:
        loadfileinner(inf)
    else:
      import sys
      reload(sys)
      sys.setdefaultencoding("utf-8")
      with codecs.open(self.filename, 'r', encoding='utf-8') as inf:
        loadfileinner(inf)

  def id2string(self, ids):
    return [self.id2stringmap.get(item, '') for item in ids]
  
  def id2vector(self, ids):
    return [self.id2vectormap.get(item, self.zero_ebm) for item in ids]
    
  def string2id(self, tokens):
    return [self.string2idmap.get(item, 0) for item in tokens]   
    
  def delvectors(self):
    self.id2vector={}
    
  def getembed(self, embed=None):
    if embed is None:
      embed = np.zeros( (self.vocabulary_size+1, self.emb_size), dtype = np.float32 )
    for k, v in self.id2vectormap.items():
      embed[k] = v
    print('Generate numpy embed: %s' % str(embed.shape))
    return embed
    
  def getembed_without_zero(self, embed=None):
    if embed is None:
      embed = np.zeros( (self.vocabulary_size, self.emb_size), dtype = np.float32 )
    for k, v in self.id2vectormap.items():
      embed[k-1] = v
    print('Generate numpy embed: %s' % str(embed.shape))
    return embed, self.emb_size

def addvocabembedding(vocab, train_embed=1, emb_type=tf.float32, emb_name="emb"):
  embed=vocab.getembed()
  return tf.Variable(embed, trainable=train_embed, dtype=emb_type, name=emb_name)
  
def addvocabembedding_with_zero(vocab, train_embed=1, emb_type=tf.float32, emb_name="emb"):
  embed, emb_size=vocab.getembed_without_zero()
  embed=tf.Variable(embed, trainable=train_embed, dtype=emb_type, name=emb_name)
  embedding_zero = tf.constant(0.0, shape=[1, emb_size], dtype=tf.float32)
  return tf.concat([embedding_zero, embed], 0)

def add_embedding_with_zero(in_size, out_size, train_embed=1, emb_type=tf.float32, emb_name="emb"):
  if (in_size < 1 or out_size < 1):
    return "Layer size ERROR"

  left  = - math.sqrt(6) / math.sqrt(in_size + out_size)
  right =   math.sqrt(6) / math.sqrt(in_size + out_size)

  embedding_zero = tf.constant(0.0, shape=[1, out_size], dtype=tf.float32)

  embed=tf.Variable(tf.random_uniform([in_size, out_size], left, right), trainable=train_embed, dtype=emb_type, name=emb_name)
  return tf.concat([embedding_zero, embed], 0)
    
def add_embedding(in_size, out_size, train_embed = 1, emb_type=tf.float32, emb_name="emb"):
  if (in_size < 1 or out_size < 1):
    return "Layer size ERROR"

  left  = - math.sqrt(6) / math.sqrt(in_size + out_size)
  right =   math.sqrt(6) / math.sqrt(in_size + out_size)

  return tf.Variable(tf.random_uniform([in_size, out_size], left, right), trainable=train_embed, dtype=emb_type, name=emb_name)

#vocab=Vocab('D:\DeepLearning\data\model2.vec.proc', 100, 1000)
#emb=addvocabembedding(vocab)
#print(str(vocab.vocabulary_size))
#print(str(vocab.id2vector([1])))
#print(str(vocab.id2string([25633,1,543,0,0,0,0,0,0,0,0,0])))
#print(str(vocal.string2id(['蔡', '</s>', '在', '有', ''])))
