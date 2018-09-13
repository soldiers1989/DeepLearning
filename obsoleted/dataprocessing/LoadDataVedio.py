#-*- coding: utf-8 -*-

'''
Utilities for Loading data.

@author: 
BinCai (caibinbupt@hotmail.com)

@references:
'''
import pandas as pd
import numpy as np
import sklearn.model_selection as skms
from sklearn import preprocessing

class LoadData(object):
  # Three files are needed in the path
  def __init__(self, datapath, dh, test_size=0.1):
    self.datapath = datapath
    self.columnsconf = datapath+'/columns2.csv'
    self.dh = dh
    self.test_size = test_size
    self.Train_DF, self.Validation_DF, self.Train_data, self.Validation_data, self.features_M = self.construct_data()

  def hdfs_to_csv(self, dtype, dh):
    columns = pd.read_csv(self.columnsconf, header=None)
    #print(columns)
    columns = columns[0].values.tolist()
    #print(len(columns))
    with open(self.datapath + '/' + dh + '/' + dtype + '_data') as f:
      lines = f.readlines()
      lines = list(map(lambda line:line.replace('\r\n', '')[1:-2].split(', '), lines))

      lines = pd.DataFrame(lines)
      for i in range(lines.shape[1]):
        if lines[i][0][0] == 'u':
          lines[i] = lines[i].apply(lambda a:a.replace("u'","").replace("'","").encode('utf-8').decode('unicode_escape'))
      lines.columns = columns
      lines.to_csv('data/'+ dh + '/' + dtype + '_data.csv', encoding='utf-8', index=False)

  def deal_vtitle(self, vtitle):
    if '微信素材' in vtitle:
      return False
    return True
  
  def read_csv(self, path, encoding = 'utf-8', usecols='', start_index=1, end_index=-1):
    with open(path, encoding=encoding) as f:
      lines = f.readlines()
      columns = lines[0].replace('\n', '').split(',')
      if start_index==1 and end_index==-1:
        data = list(map(lambda line: line.replace('\n', '').replace('None', '0').split(','), lines[1:]))
      else:
        data = list(map(lambda line: line.replace('\n', '').replace('None', '0').split(','), lines[start_index: end_index]))
      data = pd.DataFrame(data, columns = columns)
      if usecols !='':
        data.drop(list(set(columns) - set(usecols)), axis = 1, inplace = True)
      return data
  
  def get_data(self, dh):
    columns = pd.read_csv(self.columnsconf, header = None)
    columns = columns[0].values.tolist()

    data = self.read_csv('data/' + dh + '/train_data.csv', encoding = 'utf-8', start_index = 1, end_index = -1)
    data.drop_duplicates(['uin', 'vtitle'], keep='first', inplace=True)

    data = data[data['vtitle'].apply(self.deal_vtitle)]

    print(data.groupby(['label'])['label'].count())
  
    columns_old = pd.read_csv('data/columns.csv', header = None)
    columns_old = set(columns_old[0].values.tolist())

    data.drop(list(set(columns) - columns_old), axis = 1, inplace=True)
    #-set(['max_agectr', 'user_agectr_per', 'max_genderctr', 'user_genderctr_per', 'user_ageclickuv_per', 'max_ageclickuv', 'user_genderclickuv_per', 'max_genderclick_uv'])
    
    y = data['label'].values.astype(int)
    X = data.drop(['label', 'vtitle', 'uin', 'txcat', 'msgcnt', 'fopen_time', 'bizuin'], axis = 1)

    print(X.columns)

    features_map = pd.read_csv('data/features_map.csv', header=None)
    features_columns = features_map[0].values.tolist()

    df = pd.DataFrame()
    for i, feature in enumerate(features_columns):
      df.insert(i, feature, X[feature])

    df = df.astype(float)
    X = pd.DataFrame(np.nan_to_num(df.values), columns = df.columns.values.tolist())

    print(X.columns)
    print(X.shape)
    print(y.shape)

    return X, y

  def construct_data(self):
    XX, YY = self.get_data(self.dh)

    X_train, X_test, y_train, y_test = skms.train_test_split(XX, YY, test_size=self.test_size)
    T_data, V_data = {}, {}

#    std_scale = preprocessing.StandardScaler().fit(df[['Alcohol', 'Malic acid']])
#    df_std = std_scale.transform(df[['Alcohol', 'Malic acid']])
#    minmax_scale = preprocessing.MinMaxScaler().fit(df[['Alcohol', 'Malic acid']])
#    df_minmax = minmax_scale.transform(df[['Alcohol', 'Malic acid']])

    print(X_train.shape)
    print(X_test.shape)

    std_scale = preprocessing.StandardScaler().fit(X_train)
#   T_data['X'] = X_train.values.tolist()
    T_data['X'] = std_scale.transform(X_train)
    T_data['Y'] = y_train.tolist()
    
#   V_data['X'] = X_test.values.tolist()
    V_data['X'] = std_scale.transform(X_test)
    V_data['Y'] = y_test.tolist()

    return X_train, X_test, T_data, V_data, X_train.shape[1]

data = LoadData('data', '2017110520', 0.2)
print(data.features_M)
print(data.Validation_data)
print(data.Validation_data['X'].sharp())
