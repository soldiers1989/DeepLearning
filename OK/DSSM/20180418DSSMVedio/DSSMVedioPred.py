# coding: utf-8
import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from MPInputAnsyVedio import MPInputAnsyVedio
from keras.models import load_model

param = {
  'inputpath'  : 'data/',
  'predset'    : ['vedio.txt.num', 'vedio.txt2.num'],
  'dimquery'   : 5799,  #13715
  'dimdoc'     : 4858,  #12774
  'output'     : 'vedio.pred',  
  'model'      : 'D:\\DeepLearning\\data\\model.20180418152135.h5py1',
  'querymodel' : 'D:\\DeepLearning\\data\\query.model.20180418152135.h5py',
  'docmodel'   : 'D:\\DeepLearning\\data\\doc.model.20180418152135.h5py'
}

param2 = {
  'inputpath'  : '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp/src/data0417',
  'predset'    : ['vedio.txt.num', 'vedio.txt2.num'],
  'dimquery'   : 13715,
  'dimdoc'     : 12774,
  'output'     : 'vedio.pred',  
  'model'      : '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp/src/data0417',
  'querymodel' : '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp/src/data0417',
  'docmodel'   : '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp/src/data0417'
}

param=param2

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

if __name__ == '__main__':
  model = load_model(param['model'], custom_objects={'paper_loss': paper_loss,
        'paper_loss': paper_loss,
        'log2': log2,
        'euclidean': euclidean,
        'cos': cos,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'binary_PFA': binary_PFA,
        'binary_PTA': binary_PTA,
        'auc': auc }) if 'model' in param and os.path.isfile(param['model']) else None
  
  querymodel = load_model(param['querymodel'], custom_objects={'paper_loss': paper_loss,
        'paper_loss': paper_loss,
        'log2': log2,
        'euclidean': euclidean,
        'cos': cos,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'binary_PFA': binary_PFA,
        'binary_PTA': binary_PTA,
        'auc': auc }) if 'querymodel' in param and os.path.isfile(param['querymodel']) else None    
         
  docmodel = load_model(param['docmodel'], custom_objects={'paper_loss': paper_loss,
        'paper_loss': paper_loss,
        'log2': log2,
        'euclidean': euclidean,
        'cos': cos,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'binary_PFA': binary_PFA,
        'binary_PTA': binary_PTA,
        'auc': auc }) if 'docmodel' in param and os.path.isfile(param['docmodel']) else None      
            
  outfname = param['inputpath'] + '/' + param['output']
  print('open %s for write' % outfname)
  with open(outfname, 'w') as outf:
    readdata = MPInputAnsyVedio(param)
    pred_data = readdata.read_preddata_batch()
    while pred_data['L']>0:
      modelvectors = [[0]] * pred_data['L']
      if model is not None:
        modelvectors = model.predict([np.array(pred_data['Q']), np.array(pred_data['D'])])
        
      queryvectors = [[0]] * pred_data['L']
      if querymodel is not None:
        queryvectors = querymodel.predict(np.array(pred_data['Q']))
      
      docvectors = [[0]] * pred_data['L']
      if docmodel is not None:
        docvectors = docmodel.predict(np.array(pred_data['D']))

      for item in zip(pred_data['ID'], pred_data['Y'], modelvectors, queryvectors, docvectors):
        outf.write(item[0])
        outf.write('|')        
        outf.write(str(item[1]))
        outf.write('|')
        outf.write('#'.join([str(one) for one in item[2]]))
        outf.write('|')
        outf.write('#'.join([str(one) for one in item[3]]))
        outf.write('|')
        outf.write('#'.join([str(one) for one in item[4]]))
        outf.write('\n')
        
      pred_data = readdata.read_preddata_batch()  

