#-*- coding: utf-8 -*-
'''
@author:
BinCai (bincai@tencent.com)

GBDT Result
[('vmsg_bizuin_lda_1600', 105.99668455882353), ('cross_province', 87.75745230769229), ('lda_bizuin_lda_1600', 55.670075714285666), ('cross_city', 36.6166), ('ctr_uv', 30.88822170212767), ('vtitle_bizuin_lda_1600', 15.069016690000005), ('vtitle_arttopicdist', 13.682380357142858), ('vmsg_arttopicdist', 12.484228656716423), ('user_genderctr', 12.160503377777781), ('total_friend_count', 11.271676704545458), ('vtitle_vtitle_lda', 11.046604918032783), ('user_ageclickuv', 10.003312222222222), ('click_uv', 9.957362558139536), ('user_age_gender_ctr', 9.703596181818181), ('durationms', 9.449381411214958), ('lda_arttopicdist', 9.290627270270269), ('user_genderclickuv', 8.772856896551719), ('grade', 8.586026666666667), ('biz_fansnum', 8.576318142857147), ('class_bizuin_23', 8.157634561403508), ('user_agectr', 7.989551), ('agebucket', 7.9217675), ('class_vtitle_23', 7.768969099999998), ('age', 7.625786595744681), ('class_article_23', 7.582910281690142), ('vmsg_vtitle_lda', 7.49606126984127), ('play_time_avg', 7.2594665454545435), ('wordcnt', 7.151662857142857), ('lda_vtitle_lda', 6.920818225806451), ('play_time_sum', 6.880941518518519), ('show_uv', 6.6473868965517235), ('genderbucket', 5.812966666666667), ('biz_iscopyright', 4.4117425)]
Accuracy: 79.75%
Auc: 78.12%



@references:
--dim 543 --dataset 20171031data.head.ridx 20171031data.head.ridx --testset --hidden_factor 8 --layers [32,10] --keep_prob [0.8,0.5] --loss_type square_loss --activation relu --pretrain 0 --optimizer AdagradOptimizer --lr 0.05 --batch_norm 1 --verbose 1 --early_stop 1 --epoch 200

cd /mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/code; python SimpleNN.py --dim 6309 --path /mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/data/msg.org --dataset 20171031data.ridx#20171101data.ridx#20171102data.ridx --testset 20171103data.ridx --modelpath /mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/data/model --layers [128,32] --keep_prob [0.8,0.5] --loss_type log_loss --activation relu --optimizer AdamOptimizer --lr 0.05 --batch_norm 1 --verbose 1 --early_stop 0 --epoch 200
'''
import os
import sys
import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from time import time
import argparse
from LibSVMFormatInput import LoadLibSvmData
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn.metrics import roc_auc_score

#################### Arguments ####################
class SimpleNN(BaseEstimator, TransformerMixin):
  def __init__(self, dim, traindata, testdata, modelpath,
               layers, loss_type, epoch, batch_size, learning_rate,
               keep_prob, optimizer_type, batch_norm, activation_function, verbose, early_stop, random_seed=0):
    self.dim = dim
    self.traindata = traindata
    self.testdata = testdata

    # bind params to class
    self.batch_size = batch_size
    self.layers = layers
    self.loss_type = loss_type
    self.epoch = epoch
    self.random_seed = random_seed
    self.keep_prob = np.array(keep_prob)
    self.no_dropout = np.array([1 for i in range(len(keep_prob))])
    self.optimizer_type = optimizer_type
    self.learning_rate = learning_rate
    self.batch_norm = batch_norm
    self.verbose = verbose
    self.activation_function = activation_function
    self.early_stop = early_stop
    self.model_path = modelpath
    # performance of each epoch
    self.train_rmse, self.valid_rmse = [], []
    print(self.keep_prob)
    print(self.no_dropout)
    # init all variables in a tensorflow graph
    self._init_graph()

  def _init_graph(self):
    '''
    Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
    '''
    self.graph = tf.Graph()
    with self.graph.as_default():  # , tf.device('/cpu:0'):
      # Set graph level random seed
      if self.random_seed>0:
        tf.set_random_seed(self.random_seed)

      # Input data.
      with tf.name_scope('input'):
        self.train_features = tf.placeholder(tf.float32, shape=[None, None], name='train_features')  # None * features_M
        self.train_labels = tf.placeholder(tf.float32, shape=[None, 1], name='train_labels')  # None * 1
        self.dropout_keep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep')
        self.train_phase = tf.placeholder(tf.bool, name='train_phase')

      # Variables.
      self.weights = self._initialize_weights()

      # Model.
      # ________ Deep Layers __________
      self.FM = self.train_features
      for i in range(0, len(self.layers)):
        with tf.name_scope('layer_%d'%i):
          self.FM = tf.add(tf.matmul(self.FM, self.weights['layer_%d' %i]), self.weights['bias_%d'%i]) # None * layer[i] * 1
          if self.batch_norm:
            self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_%d' %i) # None * layer[i] * 1
          self.FM = self.activation_function(self.FM)
          self.FM = tf.nn.dropout(self.FM, self.dropout_keep[i]) # dropout at each Deep layer

      with tf.name_scope('output'):
        self.FM = tf.matmul(self.FM, self.weights['prediction'])     # None * 1
        self.out = tf.add(self.FM, self.weights['bias'])

      # Compute the loss.
      if self.loss_type == 'square_loss':
        self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
      elif self.loss_type == 'log_loss':
        print(self.train_labels)
        print(self.out)
        self.out = tf.sigmoid(self.out)
        self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weights=1.0, epsilon=1e-07, scope=None)

      # Optimizer.
      if self.optimizer_type == 'AdamOptimizer':
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
      elif self.optimizer_type == 'AdagradOptimizer':
        self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
      elif self.optimizer_type == 'GradientDescentOptimizer':
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
      elif self.optimizer_type == 'MomentumOptimizer':
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

      # init
      self.saver = tf.train.Saver()
      init = tf.global_variables_initializer()
      self.sess = tf.Session()
      self.sess.run(init)

      # number of params
      total_parameters = 0
      for variable in self.weights.values():
        shape = variable.get_shape() # shape is an array of tf.Dimension
        variable_parameters = 1
        for dim in shape:
          variable_parameters *= dim.value
        total_parameters += variable_parameters
      if self.verbose > 0:
        print("#params: %d" %total_parameters )  # params: 12673

  def _initialize_weights(self):
    all_weights = dict()
    num_layer = len(self.layers)

    glorot = np.sqrt(2.0 / (self.dim + self.layers[0])) # 哈维尔初始化
    all_weights['layer_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.dim, self.layers[0])), dtype=np.float32)
    print('layer_0 [input: %d, layers[0]: %d]' % (self.dim, self.layers[0]))

    for i in range(1, num_layer):
      glorot = np.sqrt(2.0 / (self.layers[i-1] + self.layers[i]))
      all_weights['layer_%d' %i] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(self.layers[i-1], self.layers[i])), dtype=np.float32)  # layers[i-1]*layers[i]
      print('layer_%d [layers[i-1]: %d, layers[i]: %d]' % (i, self.layers[i-1], self.layers[i]))

      all_weights['bias_%d' %i] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]
      print('bias_%d [1, layers[i]: %d]' % (i, self.layers[i]))

      # layer_0 [hidden_factor: 8, layers[0]: 32]   bias_0 [1, layers[0]: 32]
      # layer_1 [layers[i-1]: 32, layers[i]: 10]    bias_1 [1, layers[i]: 10]

    # prediction layer
    all_weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])), dtype=np.float32)  # 1 * layers[0]
    print('bias_0 [1, layers[0]: %d]' % self.layers[0])
    glorot = np.sqrt(2.0 / (self.layers[-1] + 1))
    all_weights['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)), dtype=np.float32)  # layers[-1] * 1
    print('prediction [1, [layers[-1]: %d]' % self.layers[-1])
    # prediction [1, [layers[-1]: 10]
    all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1

    return all_weights

  def batch_norm_layer(self, x, train_phase, scope_bn):
    bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
      is_training=True, reuse=None, trainable=True, scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
      is_training=False, reuse=True, trainable=True, scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)  #预测和训练走不同的分支
    return z

  def partial_fit(self, data):  # fit a getbatch
    #feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.dropout_keep: self.keep_prob, self.train_phase: True}
    feed_dict = {self.train_features: data['X'], self.train_labels: [[y] for y in data['Y']], self.dropout_keep: self.keep_prob, self.train_phase: True}
    loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
    return loss

  def train(self):  # fit a dataset
    # Check Init performance
    if self.verbose > 0:
      t2 = time()
      Train_data=self.traindata.read_labels_features_map_batch(self.batch_size)
      Validation_data=self.testdata.read_labels_features_map_batch(self.batch_size)

      init_train, init_train_auc = self.evaluate(Train_data)
      init_valid, init_valid_auc = self.evaluate(Validation_data)
      print("Init: \t train=%.4f, validation=%.4f train auc=%.4f, validation auc=%.4f[%.1f s]" %(init_train, init_valid, init_train_auc, init_valid_auc, time()-t2))

    log_writer = tf.summary.FileWriter(self.model_path, graph=self.sess.graph)

    for epoch in range(self.epoch):
      Train_data=self.traindata.read_labels_features_map_batch(self.batch_size)
      Validation_data=self.testdata.read_labels_features_map_batch(self.batch_size)
      t1 = time()
      loss=self.partial_fit(Train_data)
      t2 = time()

      # output validation
      train_result, train_result_auc = self.evaluate(Train_data)
      valid_result, valid_result_auc = self.evaluate(Validation_data)

      self.train_rmse.append(train_result)
      self.valid_rmse.append(valid_result)
      if self.verbose > 0 and epoch%self.verbose == 0:
        print("Epoch %d [%.1f s]\ttrain=%.4f auc=%.4f, valid=%.4f auc=%.4f, loss=%.4f [%.1f s]"
            %(epoch+1, t2-t1, train_result, train_result_auc, valid_result, valid_result_auc, loss, time()-t2))
      if self.early_stop > 0 and self.eva_termination(self.valid_rmse):
        #print "Early stop at %d based on validation result." %(epoch+1)
        break

    log_writer.close()

  def eva_termination(self, valid):
#   if self.loss_type == 'square_loss':
#     if len(valid) > 10:
#       if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
#         return True
#   else:
#     if len(valid) > 10:
#       if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
#         return True
    return False

  def evaluate(self, data):  # evaluate the results for an input set
    num_example = len(data['Y'])
    feed_dict = {self.train_features: data['X'], self.train_labels: [[y] for y in data['Y']], self.dropout_keep: self.no_dropout, self.train_phase: False}
    predictions = self.sess.run((self.out), feed_dict=feed_dict)
    y_pred = np.reshape(predictions, (num_example,))
    y_true = np.reshape(data['Y'], (num_example,))
    auc = roc_auc_score(y_true, y_pred)
    if num_example>=6:
    	print(y_pred[:6])
    	print(y_true[:6])
    else:
    	print(y_pred)
    	print(y_true)    	
    if self.loss_type == 'square_loss':
      predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
      predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
      RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
      return RMSE, auc
    elif self.loss_type == 'log_loss':
      logloss = log_loss(y_true, y_pred)
      return logloss, auc

def parse_args():
  parser = argparse.ArgumentParser(description="Run Simple NN.")
  parser.add_argument('--dim', type=int, default=6309,
            help='Dimension of data.')
  parser.add_argument('--path', nargs='?', default='/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/data/msg.org',
            help='Input data path.')
  parser.add_argument('--dataset', nargs='?', default='20171101data.ridx#20171102data.ridx',
            help='Choose a dataset.')
  parser.add_argument('--testset', nargs='?', default='20171103data.ridx',
            help='Choose a dataset.')
  parser.add_argument('--modelpath', nargs='?', default='/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/data/model',
            help='Save the model.')
  parser.add_argument('--epoch', type=int, default=200,
            help='Number of epochs.')
  parser.add_argument('--batch_size', type=int, default=128,
            help='Batch size.')
  parser.add_argument('--layers', nargs='?', default='[128, 64]',
            help="Size of each layer.")
  parser.add_argument('--keep_prob', nargs='?', default='[0.8,0.5]',
            help='Keep probability (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
  parser.add_argument('--lr', type=float, default=0.05,
            help='Learning rate.')
  parser.add_argument('--loss_type', nargs='?', default='log_loss',
            help='Specify a loss type (square_loss or log_loss).')
  parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
            help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
  parser.add_argument('--verbose', type=int, default=1,
            help='Show the results per X epochs (0, 1 ... any positive integer)')
  parser.add_argument('--batch_norm', type=int, default=0,
          help='Whether to perform getbatch normaization (0 or 1)')
  parser.add_argument('--activation', nargs='?', default='relu',
          help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
  parser.add_argument('--early_stop', type=int, default=1,
          help='Whether to perform early stop (0 or 1)')
  return parser.parse_args()

if __name__ == '__main__':
  # Data loading
  args = parse_args()

  dataset=[args.path+'/'+item for item in args.dataset.split('#')]
  testset=[args.path+'/'+item for item in args.testset.split('#')]
  traindata = LoadLibSvmData(dim=args.dim, files=dataset)
  testdata = LoadLibSvmData(dim=args.dim, files=testset)

  if args.verbose > 0:
    print("Neural FM: dataset=%s, testset=%s, dropout_keep=%s, layers=%s, loss_type=%s, #epoch=%d, getbatch=%d, lr=%.4f,optimizer=%s, batch_norm=%d, activation=%s, early_stop=%d"
        %(args.dataset, args.testset, args.keep_prob, args.layers, args.loss_type, args.epoch, args.batch_size, args.lr, args.optimizer, args.batch_norm, args.activation, args.early_stop))
  activation_function = tf.nn.relu
  if args.activation == 'sigmoid':
    activation_function = tf.sigmoid
  elif args.activation == 'tanh':
    activation_function == tf.tanh
  elif args.activation == 'identity':
    activation_function = tf.identity

  # Training
  t1 = time()
  model = SimpleNN(args.dim, traindata, testdata, args.modelpath,
                   eval(args.layers),
                   args.loss_type, args.epoch, args.batch_size, args.lr,
                   eval(args.keep_prob), args.optimizer, args.batch_norm,
                   activation_function, args.verbose, args.early_stop)

  model.train()

  # Find the best validation result across iterations
  best_valid_score = 0
  best_valid_score = max(model.valid_rmse)
  best_epoch = model.valid_rmse.index(best_valid_score)
  print ("Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
       %(best_epoch+1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], model.valid_rmse[best_epoch], time()-t1))
