# -*- coding: utf-8 -*-
'''
@author:
BinCai (bincai@tencent.com)

'''
import argparse
import datetime
import math
import os
from time import time

import numpy as np
import tensorflow as tf
from LibSVMInputEmbAnsy import LoadLibSvmDataV2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from tensorflow.contrib.tensorboard.plugins import projector


#################### Arguments ####################

class SimpleNNEmb(BaseEstimator, TransformerMixin):
  def __init__(self, args, dataset, layers, keep_prob, random_seed=0):
    self.args = args

    now = datetime.datetime.now()
    self.timestamp = now.strftime("%Y%m%d%H%M%S")

    self.dim = dataset.dim
    self.dataset = dataset

    # bind params to class
    self.batch_size = args.batch_size
    self.epoch = args.epoch

    self.layers = layers
    self.loss_type = args.loss_type
    self.keep_prob = np.array(keep_prob)
    self.no_dropout = np.array([1 for i in range(len(keep_prob))])

    l2param = eval(args.lambda_nn_l2)
    l2param.reverse()
    while len(l2param) < len(layers):
      l2param.append(0)
    l2param.reverse()
    self.lambda_nn_l2 = np.array(l2param)

    self.learning_rate = args.lr
    self.batch_norm = args.batch_norm

    self.optimizer_type = args.optimizer
    self.activation_function = tf.nn.relu
    if args.activation == 'sigmoid':
      self.activation_function = tf.sigmoid
    elif args.activation == 'tanh':
      self.activation_function = tf.tanh
    elif args.activation == 'identity':
      self.activation_function = tf.identity

    self.feature_embedding = args.feature_embedding
    self.embedding_conf = dataset.embedding_conf
    self.embedding_size = args.embedding_size
    self.embedding_init_zero = args.embedding_init_zero

    self.model_path = args.modelpath

    self.verbose = args.verbose

    self.random_seed = random_seed

    # performance of each epoch
    self.train_rmse, self.valid_rmse = [], []
    print('self.keep_prob: ' + str(self.keep_prob))
    print('self.no_dropout: ' + str(self.no_dropout))
    print('self.lambda_nn_l2: ' + str(self.lambda_nn_l2))
    print('self.lambda_pred_l2: ' + str(args.lambda_pred_l2))
    # init all variables in a tensorflow graph
    self._init_graph()

  def _init_graph(self):
    '''
    Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
    '''
    self.graph = tf.Graph()
    with self.graph.as_default():  # , tf.device('/cpu:0'):
      # Set graph level random seed
      if self.random_seed > 0:
        tf.set_random_seed(self.random_seed)

      self.lambda_pred_l2 = tf.constant(args.lambda_pred_l2, name='lambda_pred_l2')
      self.lambda_emb_l2 = tf.constant(args.lambda_emb_l2, name='lambda_emb_l2')
      self.l2_norm_init = tf.constant(0.0, name='l2_init')

      # Input data.
      with tf.name_scope('input') as scope:
        self.train_features = tf.placeholder(tf.float32, shape=[None, None],
                                             name='train_features')  # None * features_M

        self.input_emb = []
        # [(0, (11, 'bizuin', 1, 10000, 406417, 416416, 5, 2252, 2256), 6), (1, (0, 'agebucket', 1, 10, 1, 10, 7, 1, 7), 8)]
        for item in self.embedding_conf:
          self.input_emb.append(tf.placeholder(tf.int32, [None], name='input_emb_' + item[1][1]))

        self.train_labels = tf.placeholder(tf.float32, shape=[None, 1], name='train_labels')  # None * 1
        self.dropout_keep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep')
        self.nn_l2 = tf.placeholder(tf.float32, shape=[None], name='lambda_l2')
        self.train_step = tf.placeholder(tf.int32, name='train_step')
        self.train_phase = tf.placeholder(tf.bool, name='train_phase')

      # Variables.
      self.weights = self._initialize_weights()

      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
      # Model.
      # ________ Deep Layers __________
      # self.embedded_bizuin = tf.nn.embedding_lookup(self.weights['emb_0'], self.input_emb[0])
      self.embedded_lookup = []

      # concat all input
      with tf.name_scope("emb_lookup"):
        input_all = [self.train_features]
        for idx, item in enumerate(self.embedding_conf):
          embedded_lookup_item = tf.nn.embedding_lookup(self.weights['weight_emb_' + item[1][1]], self.input_emb[idx])
          self.embedded_lookup.append(embedded_lookup_item)
          input_all.append(embedded_lookup_item)
        self.NN = tf.concat(input_all, axis=1)

        self.l2_norm = self.l2_norm_init
        for idx, item in enumerate(self.embedded_lookup):
          itemloss = tf.multiply(self.lambda_emb_l2, tf.reduce_sum(tf.pow(self.embedded_lookup[idx], 2)))
          summaries.append(tf.summary.scalar("emb_l2_" + str(idx), itemloss))
          self.l2_norm += itemloss

      for i in range(0, len(self.layers)):
        layername = 'layer_%d' % i
        biasname = 'bias_%d' % i
        with tf.name_scope(layername) as scope:
          self.NN = tf.add(tf.matmul(self.NN, self.weights[layername]), self.weights[biasname])  # None * layer[i] * 1
          if self.batch_norm:
            self.NN = self.batch_norm_layer(self.NN, train_phase=self.train_phase,
                                            scope_bn='bn_%d' % i)  # None * layer[i] * 1
          self.NN = self.activation_function(self.NN)
          self.NN = tf.nn.dropout(self.NN, self.dropout_keep[i])  # dropout at each Deep layer

          layerl2 = tf.multiply(self.nn_l2[i], tf.reduce_sum(tf.pow(self.weights[layername], 2)))
          self.l2_norm += layerl2

          summaries.append(tf.summary.scalar("l2", layerl2))
          summaries.append(tf.summary.scalar("l2sum", self.l2_norm))

      with tf.name_scope('pred') as scope:
        self.NN = tf.matmul(self.NN, self.weights['prediction'])  # None * 1
        self.out = tf.add(self.NN, self.weights['bias'])

        predl2 = tf.multiply(self.lambda_pred_l2, tf.reduce_sum(tf.pow(self.weights['prediction'], 2)))
        self.l2_norm += predl2

        summaries.append(tf.summary.scalar("l2", predl2))
        summaries.append(tf.summary.scalar("l2sum", self.l2_norm))

        # Compute the loss.
        if self.loss_type == 'square_loss':
          self.loss = tf.add(tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)), self.l2_norm)
        elif self.loss_type == 'log_loss':
          self.out = tf.sigmoid(self.out)
          self.loss = tf.add(
            tf.contrib.losses.log_loss(self.out, self.train_labels, weights=1.0, epsilon=1e-07, scope=None),
            self.l2_norm)

      with tf.name_scope('opt') as scope:
        self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate,
          self.train_step, 100, 0.98, staircase=True)

        # Optimizer.
        if self.optimizer_type == 'AdamOptimizer':
          self.optimizer = tf.train.AdamOptimizer(learning_rate=self.decay_learning_rate, beta1=0.9, beta2=0.999,
                                                  epsilon=1e-8) #.minimize(self.loss)
        elif self.optimizer_type == 'AdagradOptimizer':
          self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.decay_learning_rate,
                                                     initial_accumulator_value=1e-8) #.minimize(self.loss)
        elif self.optimizer_type == 'GradientDescentOptimizer':
          self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.decay_learning_rate) #.minimize(self.loss)
        elif self.optimizer_type == 'MomentumOptimizer':
          self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.decay_learning_rate, momentum=0.95) #.minimize(self.loss)

        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        # Summary.
        for g, v in self.grads_and_vars:
          if g is not None:
            #grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            #summaries.append(grad_hist_summary)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            summaries.append(sparsity_summary)
            reduce_max_summary = tf.summary.scalar("{}/grad/max".format(v.name), tf.reduce_max(g))
            summaries.append(reduce_max_summary)
            reduce_min_summary = tf.summary.scalar("{}/grad/min".format(v.name), tf.reduce_min(g))
            summaries.append(reduce_min_summary)

        summaries.append(tf.summary.scalar("decay_learning_rate", self.decay_learning_rate))
        
      self.summary_op = tf.summary.merge(summaries)

      # init
      self.saver = tf.train.Saver()
      init = tf.global_variables_initializer()
      self.sess = tf.Session()
      self.sess.run(init)

      # number of params
      total_parameters = 0
      for variable in self.weights.values():
        shape = variable.get_shape()  # shape is an array of tf.Dimension
        variable_parameters = 1
        for dim in shape:
          variable_parameters *= dim.value
        total_parameters += variable_parameters
      if self.verbose > 0:
        print("#params: %d" % total_parameters)  # params: 12673

  def _initialize_weights(self):
    all_weights = dict()
    num_layer = len(self.layers)

    self.global_step = tf.Variable(0, name="global_step", trainable=False)

    emb_dim = 0
    with tf.name_scope('emb_lookup') as scope:
      if self.feature_embedding:
        for idx, item in enumerate(self.embedding_conf):
          weight_name = 'weight_emb_' + item[1][1]
          if self.embedding_init_zero:
            all_weights[weight_name] = tf.Variable(tf.zeros([item[2], self.embedding_size], tf.float32), name=weight_name)
          else:
            all_weights[weight_name] = tf.Variable(tf.random_uniform([item[2], self.embedding_size], -1.0, 1.0),
                                                   name=weight_name)
          emb_dim += self.embedding_size

    glorot = np.sqrt(2.0 / (self.dim + emb_dim + self.layers[0]))
    all_weights['layer_0'] = tf.Variable(
      np.random.normal(loc=0, scale=glorot, size=(self.dim + emb_dim, self.layers[0])), dtype=np.float32)
    print('layer_0 [input: %d, layers[0]: %d]' % (self.dim + emb_dim, self.layers[0]))

    for i in range(1, num_layer):
      glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
      all_weights['layer_%d' % i] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])),
        dtype=np.float32)  # layers[i-1]*layers[i]
      print('layer_%d [layers[i-1]: %d, layers[i]: %d]' % (i, self.layers[i - 1], self.layers[i]))

      all_weights['bias_%d' % i] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]
      print('bias_%d [1, layers[i]: %d]' % (i, self.layers[i]))

      # layer_0 [hidden_factor: 8, layers[0]: 32]   bias_0 [1, layers[0]: 32]
      # layer_1 [layers[i-1]: 32, layers[i]: 10]    bias_1 [1, layers[i]: 10]

    # prediction layer
    all_weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])),
                                        dtype=np.float32)  # 1 * layers[0]
    print('bias_0 [1, layers[0]: %d]' % self.layers[0])
    glorot = np.sqrt(2.0 / (self.layers[-1] + 1))
    all_weights['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)),
                                            dtype=np.float32)  # layers[-1] * 1
    print('prediction [1, [layers[-1]: %d]' % self.layers[-1])
    # prediction [1, [layers[-1]: 10]
    all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1

    return all_weights

  def batch_norm_layer(self, x, train_phase, scope_bn):
    bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                          is_training=True, reuse=None, trainable=True, scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=False, reuse=True, trainable=True, scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z

  def partial_fit(self, data, summary_writer, step):  # fit a getbatch
    if len(data['Y']) == 0:
      return 0.0
    current_step = tf.train.global_step(self.sess, self.global_step)
    feed_dict = {self.train_features: data['X'],
                 self.train_labels: [[y] for y in data['Y']],
                 self.dropout_keep: self.keep_prob,
                 self.nn_l2: self.lambda_nn_l2,
                 self.train_phase: True,
                 self.train_step: current_step }
    for idx, item in enumerate(self.embedded_lookup):
      feed_dict[self.input_emb[idx]] = [e[idx] for e in data['E']]

    loss, opt, summary = self.sess.run((self.loss, self.train_op, self.summary_op), feed_dict=feed_dict)
    summary_writer.add_summary(summary, step)
    return loss

  def train(self):  # fit a dataset
    # Check Init performance
    if self.verbose > 0:
      t2 = time()
      Train_data = self.dataset.read_traindata_batch_ansyc()
      Validation_data = self.dataset.read_testdata_batch_ansyc()

      init_train, init_train_auc = self.evaluate(Train_data)
      init_valid, init_valid_auc = self.evaluate(Validation_data)
      print("Init: \t train=%.4f, validation=%.4f train auc=%.4f, validation auc=%.4f[%.1f s]" % (
        init_train, init_valid, init_train_auc, init_valid_auc, time() - t2))

    summary_writer = tf.summary.FileWriter(self.model_path, graph=self.sess.graph)

    for item in self.embedding_conf:
      config = projector.ProjectorConfig()
      embedding_conf = config.embeddings.add()
      embedding_conf.tensor_name = 'weight_emb_' + item[1][1]
      projector.visualize_embeddings(summary_writer, config)

    t1 = time()
    for epoch in range(self.epoch):
      Train_data = self.dataset.read_traindata_batch_ansyc()
      loss = self.partial_fit(Train_data, summary_writer, epoch)

      if epoch % 10 == 0:
        t2 = time()
        # output validation
        train_result, train_result_auc = self.evaluate(Train_data)
        Validation_data = self.dataset.read_testdata_batch(self.batch_size)
        valid_result, valid_result_auc = self.evaluate(Validation_data)

        # summary info
        summary = tf.Summary(value=[
          tf.Summary.Value(tag="train_loss", simple_value=train_result),
          tf.Summary.Value(tag="train_auc", simple_value=train_result_auc),
          tf.Summary.Value(tag="test_loss", simple_value=valid_result),
          tf.Summary.Value(tag="test_auc", simple_value=valid_result_auc),
        ])

        summary_writer.add_summary(summary, epoch)

        self.train_rmse.append(train_result)
        self.valid_rmse.append(valid_result)
        if self.verbose > 0 and epoch % self.verbose == 0:
          print("Epoch %d [%.1f s]\ttrain=%.4f auc=%.4f, valid=%.4f auc=%.4f, loss=%.4f [%.1f s]"
                % (
                  epoch + 1, t2 - t1, train_result, train_result_auc, valid_result, valid_result_auc, loss,
                  time() - t2))

        t1 = time()

    self.saver.save(self.sess, os.path.join(self.model_path, "model.ckpt" + self.timestamp), epoch)

    summary_writer.close()
    if self.dataset.has_predset():
      self.pred_data()

  def pred_data(self):
    ret = self.dataset.read_preddata_batch()
    outfname = args.modelpath + '/' + args.predoutputfile

    with open(outfname, 'w') as outf:
      while ret['D'] > 0:
        num_example = ret['D']
        feed_dict = {self.train_features: ret['X'],
                     self.dropout_keep: self.no_dropout,
                     self.train_phase: False}
        for idx, item in enumerate(self.embedded_lookup):
          feed_dict[self.input_emb[idx]] = [e[idx] for e in ret['E']]

        predictions = self.sess.run((self.out), feed_dict=feed_dict)
        for k, v in zip(ret['ID'], predictions):
          outf.write(str(k) + ' ' + str(v[0]) + '\n')
        ret = self.dataset.read_preddata_batch()

  def evaluate(self, data):  # evaluate the results for an input set
    num_example = len(data['Y'])
    if num_example == 0:
      return 0.0, 0.0
    feed_dict = {self.train_features: data['X'],
                 self.train_labels: [[y] for y in data['Y']],
                 self.dropout_keep: self.no_dropout,
                 self.train_phase: False}
    for idx, item in enumerate(self.embedded_lookup):
      feed_dict[self.input_emb[idx]] = [e[idx] for e in data['E']]

    predictions = self.sess.run((self.out), feed_dict=feed_dict)
    y_pred = np.reshape(predictions, (num_example,))
    y_true = np.reshape(data['Y'], (num_example,))
    auc = roc_auc_score(y_true, y_pred)
    if num_example >= 6:
      print(y_pred[:6])
      print(y_true[:6])
    else:
      print(y_pred)
      print(y_true)
    if self.loss_type == 'square_loss':
      predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
      predictions_bounded = np.minimum(predictions_bounded,
                                       np.ones(num_example) * max(y_true))  # bound the higher values
      RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
      return RMSE, auc
    elif self.loss_type == 'log_loss':
      logloss = log_loss(y_true, y_pred)
      return logloss, auc


def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
  parser = argparse.ArgumentParser(description="Run Simple NN.")
  parser.add_argument('--data_version', type=int, default=6,
                      help='Data version')
  parser.add_argument('--inputpath', nargs='?', default='./data',
                      help='Input data path.')
  parser.add_argument('--statfile', default='statfile', required=False,
                      help='stat mapping file.')
  parser.add_argument('--remove_lowfeq', type=int, default=0,
                      help='Remove Low Frequence Feature (include)')
  parser.add_argument('--remove_feature', nargs='*', default=[], required=False,
                      help='Remove Some Feature')
  parser.add_argument('--feature_cross', type=str2bool, default=False,
                      help='cross feature.')
  parser.add_argument('--feature_embedding', nargs='*', default=[], required=False,
                      help='embedding feature.')
  parser.add_argument('--embedding_size', type=int, default=6,
                      help='embedding feature size.')
  parser.add_argument('--embedding_init_zero', type=str2bool, default=False,
                      help='embedding feature init as zero.')
  parser.add_argument('--dataset', nargs='+', default=['msg_20171031data.bincai', 'msg_20171101data.bincai'],
                      help='Choose a train dataset.')
  parser.add_argument('--testset', nargs='*', default=[],
                      help='Choose a test dataset.')
  parser.add_argument('--predset', nargs='+', default=[],
                      help='Choose a pred dataset.')
  parser.add_argument('--predoutputfile', nargs='?', default='predresult',
                      help='Choose a pred dataset.')
  parser.add_argument('--modelpath', nargs='?', default='/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/data/model',
                      help='Save the model.')
  parser.add_argument('--epoch', type=int, default=200,
                      help='Number of epochs.')
  parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size.')
  parser.add_argument('--batch_size_test', type=int, default=16384,
                      help='Data getbatch size')
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
  parser.add_argument('--lambda_nn_l2', nargs='?', default='[0.0,0.0001]',
                      help='Regulation parameter lambda for pred')
  parser.add_argument('--lambda_pred_l2', type=float, default=0.0001,
                      help='Regulation parameter lambda for NN')
  parser.add_argument('--lambda_emb_l2', type=float, default=0.0001,
                      help='Regulation parameter lambda for embedding layser')
  parser.add_argument('--shuffle_file', type=str2bool, default=True,
                      help='Suffle input file')

  return parser.parse_args()


if __name__ == '__main__':
  # Data loading
  args = parse_args()
  if args.verbose > 0:
    print(args)

  dataset = LoadLibSvmDataV2(vars(args))
  dataset.start_ansyc()

  # Training
  t1 = time()
  model = SimpleNNEmb(args, dataset, eval(args.layers), eval(args.keep_prob))
  model.train()

  dataset.stop_and_wait_ansyc()
