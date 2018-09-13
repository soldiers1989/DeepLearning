#coding: utf-8

import numpy as np
import pandas as pd

import xgboost as xgb

def dump_model_importance(model_name):
    model = xgb.Booster({'nthread':4})
    model.load_model('D:\\DeepLearning\\data\\bincai_mboost_v2.model')
    feature_map = 'D:\\DeepLearning\\data\\feature.bincai.map.txt'

    importance = model.get_score(fmap = feature_map, importance_type='gain')
    importance = sorted(importance.items(), key = lambda x: x[1], reverse = True)
    l = []
    for fname, weight in importance:
        l.append([fname, weight])
    l = pd.DataFrame(l, columns = ['fname', 'gain'])
    l.to_csv('D:\\DeepLearning\\data\\%s.importance' % model_name)

    model.dump_model('D:\\DeepLearning\\data\\%s.nicedump.txt' % model_name, fmap = feature_map, with_stats = True)

if __name__ == '__main__':
    dump_model_importance('bincai_mboost_v2')
    
#with open('data/feature.bincai.map', 'r', encoding="utf-8") as inf:
#  with open('data/feature.bincai.map.txt', 'w') as outf:
#    for i, line in enumerate(inf.readlines()): 
#      outf.write('%d %s q\n'%(i, line.strip()))