# coding: utf-8

import datetime

import keras.backend as K
import numpy as np
import tensorflow as tf
from MPInputAnsyVedio import MPInputAnsyVedio
from keras import optimizers
from keras import regularizers
from keras.layers import Dense
from keras.layers import Dot
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential

param = {
  'inputpath': 'data/',
  'dataset': ['sample.bincai.txt.num', 'sample.bincai2.txt.num'],
  'testset': ['sample.bincai.txt.num', 'sample.bincai2.txt.num'],
  'predset': '',
  'shuffle_file': True,
  'share_mlp' : True,
    
  'vocab_size': 1000,  # 12540
  'hidden_sizes': [300, 128],
  'dropout_rate': 0.5,
  'reg_rate': 0.0,
        
  'batch_size': 20,
  'steps_per_epoch': 10,
  'epochs': 2,
  'outer_epochs': 2
}

param = {
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp/src/data3/20180328.2/',
  'dataset': ['attempt_1511854667364_149204488_m_000029_0.1522330995773.num',
              'attempt_1511854667364_149204488_m_000000_0.1522330997662.num',
              'attempt_1511854667364_149204488_m_000030_0.1522330998076.num',
              'attempt_1511854667364_149204488_m_000001_0.1522330996715.num',
              'attempt_1511854667364_149204488_m_000031_0.1522330999208.num',
              'attempt_1511854667364_149204488_m_000002_0.1522331000411.num',
              'attempt_1511854667364_149204488_m_000032_0.1522330996209.num',
              'attempt_1511854667364_149204488_m_000003_0.1522330996685.num',
              'attempt_1511854667364_149204488_m_000033_0.1522330998808.num',
              'attempt_1511854667364_149204488_m_000004_0.1522330996865.num',
              'attempt_1511854667364_149204488_m_000034_0.1522330998813.num',
              'attempt_1511854667364_149204488_m_000005_0.1522330993900.num',
              'attempt_1511854667364_149204488_m_000035_0.1522331003888.num',
              'attempt_1511854667364_149204488_m_000006_0.1522330999332.num',
              'attempt_1511854667364_149204488_m_000036_0.1522330996628.num',
              'attempt_1511854667364_149204488_m_000007_0.1522330996536.num',
              'attempt_1511854667364_149204488_m_000037_0.1522330997886.num',
              'attempt_1511854667364_149204488_m_000008_0.1522331000080.num',
              'attempt_1511854667364_149204488_m_000038_0.1522330997678.num',
              'attempt_1511854667364_149204488_m_000009_0.1522330997122.num',
              'attempt_1511854667364_149204488_m_000039_0.1522330996171.num',
              'attempt_1511854667364_149204488_m_000010_0.1522330996256.num',
              'attempt_1511854667364_149204488_m_000040_0.1522330993508.num',
              'attempt_1511854667364_149204488_m_000011_0.1522330998312.num',
              'attempt_1511854667364_149204488_m_000041_0.1522330997450.num',
              'attempt_1511854667364_149204488_m_000012_0.1522330997463.num',
              'attempt_1511854667364_149204488_m_000042_0.1522330998115.num',
              'attempt_1511854667364_149204488_m_000013_0.1522330997539.num',
              'attempt_1511854667364_149204488_m_000043_0.1522330995851.num',
              'attempt_1511854667364_149204488_m_000014_0.1522330996357.num',
              'attempt_1511854667364_149204488_m_000044_0.1522330995605.num',
              'attempt_1511854667364_149204488_m_000015_0.1522330996358.num',
              'attempt_1511854667364_149204488_m_000045_0.1522330993640.num',
              'attempt_1511854667364_149204488_m_000016_0.1522330996384.num',
              'attempt_1511854667364_149204488_m_000046_0.1522330999950.num',
              'attempt_1511854667364_149204488_m_000017_0.1522330998401.num',
              'attempt_1511854667364_149204488_m_000047_0.1522330995716.num',
              'attempt_1511854667364_149204488_m_000018_0.1522330996273.num',
              'attempt_1511854667364_149204488_m_000048_0.1522331002488.num',
              'attempt_1511854667364_149204488_m_000019_0.1522330994823.num',
              'attempt_1511854667364_149204488_m_000049_0.1522330997013.num',
              'attempt_1511854667364_149204488_m_000020_0.1522330996517.num',
              'attempt_1511854667364_149204488_m_000050_0.1522331008436.num',
              'attempt_1511854667364_149204488_m_000021_0.1522330995265.num',
              'attempt_1511854667364_149204488_m_000051_0.1522330999289.num',
              'attempt_1511854667364_149204488_m_000022_0.1522331004927.num',
              'attempt_1511854667364_149204488_m_000052_0.1522330999928.num',
              'attempt_1511854667364_149204488_m_000023_0.1522330994530.num',
              'attempt_1511854667364_149204488_m_000053_0.1522330995412.num',
              'attempt_1511854667364_149204488_m_000024_0.1522330995084.num',
              'attempt_1511854667364_149204488_m_000054_0.1522330995614.num',
              'attempt_1511854667364_149204488_m_000025_0.1522330995758.num',
              'attempt_1511854667364_149204488_m_000055_0.1522330995885.num',
              'attempt_1511854667364_149204488_m_000026_0.1522330996564.num',
              'attempt_1511854667364_149204488_m_000056_0.1522330995895.num',
              'attempt_1511854667364_149204488_m_000027_0.1522330999885.num',
              'attempt_1511854667364_149204488_m_000057_0.1522330996750.num',
              'attempt_1511854667364_149204488_m_000028_0.1522330996082.num'],
  'testset': ['testattempt_1511854667364_149724370_r_000000_0.1522369981373.num',
              'testattempt_1511854667364_149724370_r_000001_0.1522369963479.num',
              'testattempt_1511854667364_149724370_r_000002_0.1522369972495.num',
              'testattempt_1511854667364_149724370_r_000003_0.1522369978278.num'],
   
  'share_mlp' : True,    
  'vocab_size': 12540,
  'hidden_sizes': [300, 128],
  'dropout_rate': 0.5,
  'reg_rate': 0.0,
       
  'batch_size': 1000,
  'steps_per_epoch': 100,
  'epochs': 10,
  'outer_epochs': 10
}

def show_layer_info(layer_name, layer_out):
  print( '[layer]: %s\t[shape]: %s \n' % (layer_name,str(layer_out.get_shape().as_list())) )

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
    return 0.0015
  elif epoch >= 10 and epoch < 20:
    return 0.001
  else:
    return 0.0005

def train_data_generator(dateset):
  while True:
    train_data = readdata.read_traindata_batch_ansyc()
    x_train, y_train = [np.array(train_data['Q']), np.array(train_data['D'])], np.array(train_data['Y']).transpose()
    yield x_train, y_train

def validation_data_generator(dateset):
  while True:
    valid_data = readdata.read_testdata_batch(1024)
    x_valid, y_valid = [np.array(valid_data['Q']), np.array(valid_data['D'])], np.array(valid_data['Y']).transpose()
    yield x_valid, y_valid

def create_model():
  title = Input(name='title', shape=(param['vocab_size'],))#, sparse=True)
  show_layer_info('Input', title)
  doc = Input(name='doc', shape=(param['vocab_size'],))#, sparse=True)
  show_layer_info('Input', doc)

  def mlp_work(input_dim):
    seq = Sequential()
    num_hidden_layers = len(param['hidden_sizes'])
    if num_hidden_layers == 1:
      seq.add(Dense(param['hidden_sizes'][0], input_shape=(input_dim,), activity_regularizer=regularizers.l2(param['reg_rate'])))
    else:
      seq.add(Dense(param['hidden_sizes'][0], activation='tanh', input_shape=(input_dim,), activity_regularizer=regularizers.l2(param['reg_rate'])))
      for i in range(num_hidden_layers-2):
        seq.add(Dense(param['hidden_sizes'][i+1], activation='tanh', activity_regularizer=regularizers.l2(param['reg_rate'])))
        seq.add(Dropout(rate=param['dropout_rate']))
      seq.add(Dense(param['hidden_sizes'][num_hidden_layers-1], activity_regularizer=regularizers.l2(param['reg_rate'])))
      seq.add(Dropout(rate=param['dropout_rate']))

    return seq

  mlp = mlp_work(param['vocab_size'])
  rq = mlp(title)
  show_layer_info('MLP', rq)
  rd = mlp(doc)
  show_layer_info('MLP', rd)

  out_ = Dot( axes= [1, 1], normalize=True)([rq, rd])
  show_layer_info('Dot', out_)

  model = Model(inputs=[title, doc], outputs=[out_])
  model.summary()
  return mlp, model

#_________________________________________________________________________________________________
#Layer (type)                    Output Shape         Param #     Connected to
#==================================================================================================
#title (InputLayer)              (None, 1000)         0
#__________________________________________________________________________________________________
#doc (InputLayer)                (None, 1000)         0
#__________________________________________________________________________________________________
#sequential_1 (Sequential)       (None, 128)          355340      title[0][0]
#                                                                 doc[0][0]
#__________________________________________________________________________________________________
#dot_1 (Dot)                     (None, 1)            0           sequential_1[1][0]
#                                                                 sequential_1[2][0]
#__________________________________________________________________________________________________
#dense_4 (Dense)                 (None, 2)            4           dot_1[0][0]
#==================================================================================================
#Total params: 355,344
#Trainable params: 355,344
#Non-trainable params: 0
#__________________________________________________________________________________________________


def paper_loss(y_true, y_pred):
  return y_true * 1 / 4 * K.square((1 - y_pred)) + (1 - y_true) * K.square(K.maximum(y_pred, 0))

def train(dateset):
  mlp, model = create_model()
  model.compile(optimizer=optimizers.Nadam(lr=0.0005), loss=paper_loss, metrics=[accuracy, precision, recall, auc])
#  model.compile(optimizer=optimizers.Nadam(lr=0.0005), loss=losses.binary_crossentropy, metrics=[accuracy, precision, recall, auc])
  
  for it in range(param['outer_epochs']):
    model.fit_generator(generator=train_data_generator(readdata),
                        steps_per_epoch=param['steps_per_epoch'],
                        # Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch.
                        epochs=param['epochs'],
                        verbose=2,  # 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
                        validation_data=validation_data_generator(readdata),
                        validation_steps=10)
    
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    modelfilepath = param['inputpath'] + "dssm." + ts + ".h5py"
    model.save(modelfilepath)
    modelfilepath = param['inputpath'] + "dssm.mlp." + ts + ".h5py"
    mlp.save(modelfilepath)

if __name__ == '__main__':
  param['dim']=param['vocab_size']
  readdata = MPInputAnsyVedio(param)
  readdata.start_ansyc()

  train(readdata)

  readdata.stop_and_wait_ansyc()


