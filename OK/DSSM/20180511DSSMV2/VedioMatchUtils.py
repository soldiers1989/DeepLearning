#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import math
import sys

def calculate_y (x, w, b, keep_prob=1.0):
  y = tf.matmul(x, w) + b 
  y = tf.nn.dropout(y, keep_prob)
  return tf.nn.relu(y)
  #return tf.nn.softsign(y)

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

# Read Embedding File
def read_embedding(filename, index=1):
  embed = {}
  for line in open(filename):
    line = line.strip().split('\t')
    vector = line[index+1].split()[1:]
    if len(vector) != 300:
      continue
    embed[int(line[index])-1] = map(float, vector)
  print('[%s]\n\tEmbedding size: %d' % (filename, len(embed)))
  return embed

# Convert Embedding Dict 2 numpy array
def convert_embed_2_numpy(embed_dict, max_size=0, embed=None):
  feat_size = len(embed_dict[list(embed_dict.keys())[0]])
  if embed is None:
    embed = np.zeros( (max_size, feat_size), dtype = np.float32 )
  for k in embed_dict:
    embed[k] = np.array(embed_dict[k])
  print('Generate numpy embed: %s'%str(embed.shape))
  return embed

def init_embedding(vocab_size, embed_size, embed_path):
  # collect embedding
  left  = - math.sqrt(6) / math.sqrt(vocab_size + embed_size)
  right =   math.sqrt(6) / math.sqrt(vocab_size + embed_size)
  embed_dict = read_embedding(filename=embed_path, index=1)
  #_PAD_ = 0
  #embed_dict[_PAD_] = np.zeros((config.embed_size, ), dtype=np.float32)
  embed = np.float32(np.random.uniform(left, right, [vocab_size, embed_size]))
  embed = convert_embed_2_numpy(embed_dict, embed = embed)

  print('[Embedding] Embedding Load Done.')
  return embed

def add_embedding(in_size, out_size, embed_path='', train_embed = 1, emb_type=tf.float32, emb_name="emb"):
  if (in_size < 1 or out_size < 1):
    return "layer size ERROR"

  left  = - math.sqrt(6) / math.sqrt(in_size + out_size)
  right =   math.sqrt(6) / math.sqrt(in_size + out_size)

  if embed_path == '':
    return tf.Variable(tf.random_uniform([in_size, out_size], left, right), trainable=train_embed, dtype=emb_type, name=emb_name)
  else:
    embed = init_embedding(in_size, out_size, embed_path)
    return tf.Variable(embed, trainable=train_embed, dtype=emb_type, name=emb_name)
    
def add_embedding_with_zero(in_size, out_size, embed_path='', train_embed = 1, emb_type=tf.float32, emb_name="emb"):
  if (in_size < 1 or out_size < 1):
    return "layer size ERROR"

  left  = - math.sqrt(6) / math.sqrt(in_size + out_size)
  right =   math.sqrt(6) / math.sqrt(in_size + out_size)
  
  embedding_zero = tf.constant(0.0, shape=[1, out_size], dtype=tf.float32)

  if embed_path == '':
    embed=tf.Variable(tf.random_uniform([in_size, out_size], left, right), trainable=train_embed, dtype=emb_type, name=emb_name)
    return tf.concat([embedding_zero, embed], 0)
  else:
    embed = init_embedding(in_size, out_size, embed_path)
    embed1 = tf.Variable(embed, trainable=train_embed, dtype=emb_type, name=emb_name)
    return tf.concat([embedding_zero, embed1], 0)

def compute_triplet_loss(title_feature, pos_feature, neg_feature, margin="0.1"):
  sim_pos = tf.reduce_sum(tf.multiply(title_feature, pos_feature), 1)
  sim_neg = tf.reduce_sum(tf.multiply(title_feature, neg_feature), 1)
  loss = tf.maximum(0., sim_neg - sim_pos + margin)
  sim_loss = 1 - sim_pos
  return tf.reduce_mean(loss)

def get_acc(title_z1_out, pos_z1_out, neg_z1_out):
  sim_pos = np.sum(np.multiply(title_z1_out, pos_z1_out), 1)
  sim_neg = np.sum(np.multiply(title_z1_out, neg_z1_out), 1)
  pos_sub_neg = sim_pos - sim_neg
  total_num = title_z1_out.shape[0]
  acc_num = 0 
  for i in range(total_num):
    diff_sim = pos_sub_neg[i]
    if (diff_sim > 0): 
      acc_num = acc_num + 1 
  acc = acc_num * 1.0 / total_num
  return acc
  
def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))
 
def get_accprecrecauc(title_z1_out, pos_z1_out, neg_z1_out, margin="0.1"):
  sim_pos = np.sum(np.multiply(title_z1_out, pos_z1_out), 1)
  sim_neg = np.sum(np.multiply(title_z1_out, neg_z1_out), 1)
  loss = np.maximum(0., sim_neg - sim_pos + margin)
  print('-'*20)
  print('sim_pos %s' % str(sim_pos) )
  print('sim_neg %s' % str(sim_neg) )
  print('loss %s' % str(loss) )
  pos_sub_neg = sim_pos - sim_neg
  total_num = title_z1_out.shape[0]
  acc_num = 0 
  for i in range(total_num):
    diff_sim = pos_sub_neg[i]
    if (diff_sim > 0): 
      acc_num = acc_num + 1 
  acc = acc_num * 1.0 / total_num
  
  y_true=[1.0]*title_z1_out.shape[0]
  y_true.extend([0.0]*title_z1_out.shape[0])
  y_pred=list(sigmoid(sim_pos))
  y_pred.extend(list(sigmoid(sim_neg)))
  auc = roc_auc_score(y_true, y_pred) 
  y_pred = [ 1 if item>0.5 else 0 for item in y_pred ]
  y_true = [ 1 if item>0.5 else 0 for item in y_true ]
  return acc, precision_score(y_true, y_pred, average='binary'), recall_score(y_true, y_pred, average='binary'), auc
  
def print_model_data(lv, user_z1_out, pos_z1_out, neg_z1_out, pos_i, neg_i, user_i):
#  print('lv %f'%lv)
#  print('user_z1_out shape %s'%str(user_z1_out.shape))
#  print('pos_z1_out shape %s'%str(pos_z1_out.shape))
#  print('neg_z1_out shape %s'%str(neg_z1_out.shape))
#  print('pos_i shape %s'%str(pos_i.shape))
#  print('neg_i shape %s'%str(neg_i.shape))
#  print('user_i shape %s'%str(user_i.shape))
  
  for ii in range(1):
  #for ii in range(user_z1_out.shape[0]):
    print('*'*20+str(ii))
    print( 'user_z1_out[%d] %s' % (ii, str(user_z1_out[0])) )
    print( 'pos_z1_out[%d] %s' % (ii, str(pos_z1_out[0])) )
    print( 'neg_z1_out[%d] %s' % (ii, str(neg_z1_out[0])) )
    print( 'pos_i[%d] %s' % (ii, str(pos_i[0])) )
    print( 'neg_i[%d] %s' % (ii, str(neg_i[0])) )
    print( 'user_i[%d] %s' % (ii, str(user_i[0])) )

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

def load_data(data_dir="../data/", file_name="to_train.lst"):
  file=data_dir + file_name
  try:
    inf = open(file, "r")
    
  except:
    print("not file excepttion input file not find")
    return 

  line  = inf.readline()
  res = []
  while line:
    line = line.strip()
    if not line:
      break
    line_ = line.split(";")
    if (len(line_) < 3): 
      line = inf.readline()
      continue
    title = line_[0].strip()
    pos   = line_[1].strip()
    neg   = line_[2].strip()
    if not title or not pos or not neg:
      line = inf.readline()
      continue
    res.append(line_)
    line =  inf.readline()
  inf.close()
  return res

def get_term_id_dict(data_dir="./data/", file_name="term_id.dict"):    
  file=data_dir + file_name
  try:
    inf = open(file, "r")
  except: 
    print("cant not find term id dict")
    return   
  term_dict = {}
  line  = inf.readline()   
  while line:  
    line = line.strip()  
    if not line:
      break
    line_ = line.split("\t")
    if (len(line_) < 2): 
      line = inf.readline()
      continue
    term_dict[line_[0]] = line_[1]
    line =  inf.readline()
  inf.close()
  return term_dict

def to_ids(terms, dict_tmp):
  if (len(dict_tmp) < 1): 
    return ""  
  res = []  
  for t in terms:
    t =  t.encode("utf8")
    if (not dict_tmp.has_key(t)):
      continue
    res.append(int (dict_tmp[t]))
  return res

def to_string(vec):
  res = ""
  for v in vec:
    res = res + str("%.5f"%v) + " " 
  res = res.strip()
  return res

def output_res(vec_list , title_list, outvec_file):
  if (vec_list.shape[0] != len(title_list)):
    print("ERROR, length of title_list not equal length of vec_list")
    return
  for i in range(len(title_list)):
    title = title_list[i].strip()
    vec   = to_string(vec_list[i])
    if (not title or not vec):
      continue
    outvec_file.write(title + "\t" + vec + '\n') 
  return

