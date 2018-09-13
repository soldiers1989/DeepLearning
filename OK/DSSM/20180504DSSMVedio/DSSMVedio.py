# coding: utf-8

import datetime

import keras.backend as K
import numpy as np
import tensorflow as tf
from MPInputAnsyVedio import MPInputAnsyVedio
from keras import losses
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense
from keras.layers import Dot
from keras.layers import Input
from keras.models import Model

param = {
  'inputpath': 'data/',
  'dataset': ['vedio.txt.num', 'vedio.txt2.num'],
  'testset': ['vedio.txt.num', 'vedio.txt2.num'],
  'predset': '',
  'shuffle_file': True,
  'batch_size': 10,
  'dimquery': 5799,  # 12540
  'dimdoc':   4858,  # 12540
  'vec_size': 128,
  'steps_per_epoch':10,
  'epochs': 10,
  'outer_epochs': 10
}

param2 = {
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp/src/data0417/',
  'dataset': ['train0', 'train1', 'train2', 'train3', 'train4', 
               'train5', 'train6', 'train7', 'train8', 'train9'],
  'testset': ['test0', 'test1', 'test2', 'test3', 'test4', 
               'test5', 'test6', 'test7', 'test8', 'test9'],
  'predset': '',
  'shuffle_file': True,
  'batch_size': 1024,
  'dimquery': 13715,
  'dimdoc':   12774,
  'vec_size': 128,
  'steps_per_epoch':100,
  'epochs': 100,
  'outer_epochs': 20
}

#param=param2

def log2(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator

def euclidean(x1, x2):
  return np.sqrt(np.sum((x1 - x2) ** 2))

def cos(x1, x2):
  return np.dot(x1, x2) / (np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2)))

def jun_loss(y_true, y_pred):
  # return -K.log(K.sigmoid(y_true*y_pred))
  return y_true * 1 / 4 * K.square((1 - y_pred)) + (1 - y_true) * log2(1 + K.maximum(y_pred, 0))

def paper_loss(y_true, y_pred):
  #return y_true * 1 / 4 * K.square((1 - y_pred)) + (1 - y_true) * K.square(K.maximum(y_pred, 0))
  y_pred = K.squeeze(y_pred, axis=1)
  y_true = K.squeeze(y_true, axis=1)
  loss = K.mean( y_true * 1/4 * K.square((1 - y_pred)) + (1 - y_true) * K.square(K.maximum(y_pred, 0)) )
  return loss

def accuracy(y_true, y_pred):
  return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))

def precision(y_true, y_pred):
  c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
  c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
  p = c1 / (c2 + K.epsilon())
  return p

def recall(y_true, y_pred):
  c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
  c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
  r = c1 / (c3 + K.epsilon())
  return r

def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
  y_pred = K.cast(y_pred >= threshold, 'float32')
  N = K.sum(1 - y_true)
  FP = K.sum(y_pred - y_pred * y_true)
  return FP / (N + K.epsilon())

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
  y_pred = K.cast(y_pred >= threshold, 'float32')
  P = K.sum(y_true)
  TP = K.sum(y_pred * y_true)
  return TP / (P + K.epsilon())

def auc(y_true, y_pred):
  ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
  pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
  pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
  binSizes = -(pfas[1:] - pfas[:-1])
  s = ptas * binSizes
  return K.sum(s, axis=0)

def get_lr(epoch):
  if epoch < 10:
    return 0.001
  elif epoch >= 10 and epoch < 20:
    return 0.0005
  else:
    return 0.0002

def create_dnn_model(vocab_size, modelname):
  ninput = Input(shape=(vocab_size,))
  x = Dense(param['vec_size'], activation='tanh')(ninput)
  y = Dense(param['vec_size'], activation='tanh')(x)
  model = Model(ninput, y, name='%s_%s' % (modelname, vocab_size))
  return model

def model_framework_bincai():
  query_input = Input(shape=(param['dimquery'],), name='query_input')
  query_model = create_dnn_model(param['dimquery'], 'query_model')
  query_encode = query_model(query_input)

  doc_input = Input(shape=(param['dimdoc'],), name='doc_input')
  doc_model = create_dnn_model(param['dimdoc'], 'doc_model')
  doc_encode = doc_model(doc_input)

  #out = Dot( axes= [1, 1], normalize=True)([query_encode, doc_encode])
  out = Dot(axes=1, normalize=True)([query_encode, doc_encode])
  model = Model([query_input, doc_input], [out])

  query_model.summary()
  doc_model.summary()
  model.summary()

  return model, query_model, doc_model

def train_data_generator(dateset):
  while True:
    train_data = readdata.read_traindata_batch_ansyc()
    x_train, y_train = [np.array(train_data['Q']), np.array(train_data['D'])], np.array(train_data['Y']).transpose()
    #print('len(x_train) %d x_train[0].shape %s x_train[1].shape %s y_train.shape %s' % (len(x_train), x_train[0].shape, x_train[1].shape, y_train.shape))
    yield x_train, y_train

def validation_data_generator(dateset):
  while True:
    valid_data = readdata.read_testdata_batch(1024)
    x_valid, y_valid = [np.array(valid_data['Q']), np.array(valid_data['D'])], np.array(valid_data['Y']).transpose()
    #print('len(x_valid) %d x_valid[0].shape %s x_valid[1].shape %s y_valid.shape %s' % (len(x_valid), x_valid[0].shape, x_valid[1].shape, y_valid.shape))
    yield x_valid, y_valid

def train3(dateset):
  lrdecay = LearningRateScheduler(get_lr)
  model, query_model, doc_model = model_framework_bincai()

  #model.compile(optimizer=optimizers.Nadam(lr=0.001), loss=paper_loss, metrics=[accuracy, precision, recall, auc])
  model.compile(optimizer=optimizers.Nadam(lr=0.0002), loss=losses.binary_crossentropy, metrics=[accuracy, precision, recall, auc])

  #  train_data = readdata.read_traindata_batch_ansyc()
  #  x_train = [ np.array(train_data['Q']), np.array(train_data['D']) ]
  #  y_train = (np.array(train_data['Y']).transpose())
  #  print('*'*20)
  #  print('len(x_train) %d x_train[0].shape %s x_train[1].shape %s y_train.shape %s' % (len(x_train), x_train[0].shape, x_train[1].shape, y_train.shape))

#  filepath = param['inputpath'] + "weights-improvement-{epoch:02d}.h5py"
#  checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
#  callbacks_list = [checkpoint]
  for it in range(param['outer_epochs']):
    model.fit_generator(generator=train_data_generator(readdata),
                        steps_per_epoch=param['steps_per_epoch'],
                        # Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch.
                        epochs=param['epochs'],
                        verbose=2,  # 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
                        # callbacks=callbacks_list,  # 其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用
                        validation_data=validation_data_generator(readdata),
                        validation_steps=10)

    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    modelfilepath = param['inputpath'] + "model." + ts + ".h5py"
    model.save(modelfilepath)
    modelfilepath = param['inputpath'] + "query.model." + ts + ".h5py"
    query_model.save(modelfilepath)
    modelfilepath = param['inputpath'] + "doc.model." + ts + ".h5py"
    doc_model.save(modelfilepath)

if __name__ == '__main__':
  readdata = MPInputAnsyVedio(param)
  readdata.start_ansyc()

  train3(dateset=readdata)

  #  try:
  #   train3(param['dim'], dateset=readdata)
  #  except:
  #   print('Exit by exception')
  readdata.stop_and_wait_ansyc()

