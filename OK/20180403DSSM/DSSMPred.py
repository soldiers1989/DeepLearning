# coding: utf-8

import keras.backend as K
import tensorflow as tf
from MPInputAnsy import MPInputAnsy
from keras.layers import Dense
from keras.layers import Dot
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
import numpy as np

param = {
  'model' : 'D:\DeepLearning\data\weights-improvement-01.h5py',
  'vocab' : ''
}

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

def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
  y_pred = K.cast(y_pred >= threshold, 'float32')
  N = K.sum(1 - y_true)
  FP = K.sum(y_pred - y_pred * y_true)
  return FP/(N + K.epsilon())

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
  y_pred = K.cast(y_pred >= threshold, 'float32')
  P = K.sum(y_true)
  TP = K.sum(y_pred * y_true)
  return TP/(P + K.epsilon())

def auc(y_true, y_pred):
  ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
  pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
  pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
  binSizes = -(pfas[1:]-pfas[:-1])
  s = ptas*binSizes
  return K.sum(s, axis=0)

def get_lr(epoch):
  if epoch < 10:
    return 0.0015
  elif epoch >= 10 and epoch < 20:
    return 0.001
  else:
    return 0.0005

def create_dnn_model(text_length):
  ninput = Input(shape=(text_length,))
  x = Dense(param['vec_size'], activation='relu')(ninput)
  y = Dense(param['vec_size'], activation='tanh')(x)
  model = Model(ninput, y, name='share_model_len%s' % (text_length))
  return model

def model_framework_bincai(vocab_size):
  content_model = create_dnn_model(vocab_size)
  title_model = content_model

  input1 = Input(shape=(vocab_size,), name='article1_input')
  input2 = Input(shape=(vocab_size,), name='article2_input')
  encode_1 = title_model(input1)
  encode_2 = content_model(input2)

  # out = Dot( axes= [1, 1], normalize=True)([encode_1, encode_2])
  out = Dot(axes=1, normalize=True)([encode_1, encode_2])
  model = Model([input1, input2], [out])

  content_model.summary()
  model.summary()

  return model, content_model, content_model

if __name__ == '__main__':
#  readdata = MPInputAnsy(param)
#  readdata.start_ansyc()
#
#  train3(param['dim'], dateset=readdata)
  base_model = load_model(param['model'], custom_objects={'paper_loss': paper_loss,
  	  	'paper_loss': paper_loss,
  	  	'log2': log2,
  	  	'euclidean': euclidean,
  	  	'cos': cos,
  	  	'accuracy': accuracy,
  	  	'precision': precision,
  	  	'recall': recall,	  
  	  	'binary_PFA': binary_PFA,	  
  	  	'binary_PTA': binary_PTA,	  
  	  	'auc': auc })
  model = Model(inputs=base_model.input, outputs=base_model.get_layer('share_model_len1000').get_output_at(0) )
  	  		
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#input_1 (InputLayer)         (None, 12540)             0         
#_________________________________________________________________
#dense_1 (Dense)              (None, 128)               1605248   
#_________________________________________________________________
#dense_2 (Dense)              (None, 128)               16512     
#=================================================================
#Total params: 1,621,760
#Trainable params: 1,621,760
#Non-trainable params: 0
#_________________________________________________________________
#____________________________________________________________________________________________________
#Layer (type)                     Output Shape          Param #     Connected to                     
#====================================================================================================
#article1_input (InputLayer)      (None, 12540)         0                                            
#____________________________________________________________________________________________________
#article2_input (InputLayer)      (None, 12540)         0                                            
#____________________________________________________________________________________________________
#share_model_len12540 (Model)     (None, 128)           1621760     article1_input[0][0]             
#                                                                   article2_input[0][0]             
#____________________________________________________________________________________________________
#dot_1 (Dot)                      (None, 1)             0           share_model_len12540[1][0]       
#                                                                   share_model_len12540[2][0]       
#====================================================================================================
#Total params: 1,621,760
#Trainable params: 1,621,760
#Non-trainable params: 0

#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#input_1 (InputLayer)         (None, 1000)              0         
#_________________________________________________________________
#dense_1 (Dense)              (None, 128)               128128    
#_________________________________________________________________
#dense_2 (Dense)              (None, 128)               16512     
#=================================================================
#Total params: 144,640
#Trainable params: 144,640
#Non-trainable params: 0
#_________________________________________________________________
#__________________________________________________________________________________________________
#Layer (type)                    Output Shape         Param #     Connected to                     
#==================================================================================================
#article1_input (InputLayer)     (None, 1000)         0                                            
#__________________________________________________________________________________________________
#article2_input (InputLayer)     (None, 1000)         0                                            
#__________________________________________________________________________________________________
#share_model_len1000 (Model)     (None, 128)          144640      article1_input[0][0]             
#                                                                 article2_input[0][0]             
#__________________________________________________________________________________________________
#dot_1 (Dot)                     (None, 1)            0           share_model_len1000[1][0]        
#                                                                 share_model_len1000[2][0]        
#==================================================================================================
#Total params: 144,640
#Trainable params: 144,640
#Non-trainable params: 0
	
