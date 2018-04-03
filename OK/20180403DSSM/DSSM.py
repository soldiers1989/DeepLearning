# coding: utf-8

import keras.backend as K
import tensorflow as tf
from MPInputAnsy import MPInputAnsy
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense
from keras.layers import Dot
from keras.layers import Input
from keras.models import Model

import config
from config import *


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
  return y_true * 1 / 4 * K.square((1 - y_pred)) + (1 - y_true) * K.square(K.maximum(y_pred, 0))

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

def get_lr(epoch):
  if epoch < 10:
    return 0.0015
  elif epoch >= 10 and epoch < 20:
    return 0.001
  else:
    return 0.0005

def create_dnn_model(text_length):
  ninput = Input(shape=(text_length,))
  x = Dense(config.vec_size, activation='relu')(ninput)
  y = Dense(config.vec_size, activation='tanh')(x)
  model = Model(ninput, y, name='share_model_len%s' % (text_length))
  return model  

def model_framework_bincai(vocab_size): 
  content_model = create_dnn_model(vocab_size)
  title_model = content_model

  input1 = Input(shape=(vocab_size,), name='article1_input')
  input2 = Input(shape=(vocab_size,), name='article2_input')
  encode_1 = title_model(input1)
  encode_2 = content_model(input2)

  #out = Dot( axes= [1, 1], normalize=True)([encode_1, encode_2])
  out = Dot( axes=1, normalize=True )([encode_1, encode_2])
  model = Model([input1, input2], [out])

  content_model.summary()
  model.summary()

  return model, content_model, content_model

def train3(vocab_size, dateset):    
  lr = LearningRateScheduler(get_lr)
  model, content_model, content_model = model_framework_bincai(vocab_size)

  parallel_model = model
  parallel_model.compile(optimizer=optimizers.Nadam(lr=0.002), loss=paper_loss, metrics=[accuracy, precision, recall])

  train_data = readdata.read_traindata_batch_ansyc()
  valid_data = readdata.read_testdata_batch(1024)
  x_train = [ np.array(train_data['Q']), np.array(train_data['D']) ]
  y_train = (np.array(train_data['Y']).transpose())
  print('*'*20)
  print('len(x_train) %d x_train[0].shape %s x_train[1].shape %s y_train.shape %s' % (len(x_train), x_train[0].shape, x_train[1].shape, y_train.shape))
  # len(x_train) 2 x_train[0].shape (100, 1000) y_train.shape (100,)
  # x1_text.shape (50943, 30)    y_train.shape (50943,)

  x_valid = [ np.array(valid_data['Q']), np.array(valid_data['D']) ]
  y_valid = ( np.array(valid_data['Y']).transpose() )
  print('len(x_valid) %d x_valid[0].shape %s x_valid[1].shape %s y_valid.shape %s' % ( len(x_valid), x_valid[0].shape, x_valid[1].shape, y_valid.shape))

  parallel_model.fit(x_train, y_train, epochs=config.epochs, verbose=2, batch_size=config.batch_size,
                     validation_data=[x_valid, y_valid], shuffle=True)

if __name__ == '__main__':
  param = { 'inputpath':'data/',
            'dataset':['sample.bincai.txt.num', 'sample.bincai2.txt.num'],
            'testset':['sample.bincai.txt.num', 'sample.bincai2.txt.num'],
            'predset':'',
            'shuffle_file': True,
            'batch_size': 100,
            'dim': 1000 #12540
          } 
          
  readdata = MPInputAnsy(param)
  readdata.start_ansyc()
  
  train3(param['dim'], dateset=readdata) 
  
#  try:  
#   train3(param['dim'], dateset=readdata)
#  except:
#   print('Exit by exception')
  readdata.stop_and_wait_ansyc()

