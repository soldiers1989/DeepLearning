#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import datetime
import os
import time
import tensorflow as tf

import TFBCUtils
from TFBCUtils import Vocab
from ToutiaoSimInput import ToutiaoSimInput
from ToutiaoSim import ToutiaoSim

param = {
  'inputpath': 'data/',
  'modelpath': 'model/',
  'dataset': ['toutiao.txt', 'toutiao.txt'],
  'testset': ['toutiao.txt'],
  'predset': [],

  'shuffle_file': True,
  'batch_size': 16,
  'batch_size_test': 16,
  'test_batch': 100,
  'save_batch': 500,
  'total_batch': 1000,
  'decay_steps': 1000,
  'keep_prob': 0.5,
  'grad_clip': 1.5,
  'margin': 0.2,
  'lr': 0.0001,
    
  'emb_size': 100,
  'layers': 2,
  'titlemax_size': 20,
  'contentmax_size': 200,

  'vocab': 'data/model.vec.num',
  'vocab_size': 1000,
  'kernel_sizes': [1, 2],
  'filters': 2
}

param2 = {
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/toutiao/20181007/train/',
  'modelpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/toutiao/20181007/model/',

  'dataset': ['part-00000','part-00002','part-00004','part-00006','part-00008','part-00010','part-00012','part-00014','part-00016','part-00018',\
              'part-00001','part-00003','part-00005','part-00007','part-00009','part-00011','part-00013','part-00015','part-00017','part-00019'],
  'testset': [],
  'predset': [],

  'batch_size': 64,
  'batch_size_test': 4096,
  'test_batch': 100,
  'save_batch': 5000,
  'total_batch': 400000,
  'decay_steps': 5000,
  'margin': 0.3,
  'keep_prob': 0.5,

  'vocab': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/toutiao/fasttext/model.vec.num',
  'vocab_size': 333294,
  'kernel_sizes': [1, 2, 3, 4],
  'filters': 200
}

param.update(param2)

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():  # --tasks dailytrain --inputpath data/ --modelpath model/
	# --inputpath /mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/toutiao/20181007/article/ --predset part-00000 part-00001 part-00002 --tasks predmsg --ckpt /mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/toutiao/20181007/model/toutiao-model-20181009112350-25000
	
	# python ToutiaoSimRun.py --inputpath /mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/toutiao/20181007/mptmp/ --predset part-00000 part-00002 part-00004 part-00006 part-00001 part-00003 part-00005 part-00007 part-00009 part-00008 --tasks predmsg --ckpt /mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/toutiao/20181007/model/toutiao-model-20181009112350-60000
  parser = argparse.ArgumentParser(description="Run Vedio Classify NN.")
  parser.add_argument('--tasks', nargs='+', default=['train'],
                      # ['train', 'dailytrain', 'predmsg'],
                      help='Using pred function.')
  parser.add_argument('--inputpath', nargs='?', default='data/',
                      help='Input data path.')
  parser.add_argument('--dataset', nargs='+', default=['toutiao.txt', 'toutiao.txt'],
                      help='Choose a train dataset.')
  parser.add_argument('--modelpath', nargs='?', default='model/',
                      help='Model output path.')
  parser.add_argument('--predset', nargs='+', default=['toutiaoati.txt'],
                      help='Choose a vedio pred dataset.')
  parser.add_argument('--predsetoutput', nargs='?', default='msg.pred',
                      help='Choose a msg pred file.')
  parser.add_argument('--ckpt', nargs='?', default='D:\\DeepLearning\\model\\toutiao-model-20180926203149-500',
                      help='Path to save the model.')
  return parser.parse_args()


if __name__ == "__main__":  
  args = parse_args()
  tasks = set(args.tasks)
  print(str(tasks))

  if 'train' in tasks or 'dailytrain' in tasks:
    if 'dailytrain' in tasks:
      param['inputpath'] = args.inputpath
      param['modelpath'] = args.modelpath

    readdata = ToutiaoSimInput(param)
    readdata.start_ansyc()
    vocab = Vocab(param['vocab'], param['emb_size'])
    model = ToutiaoSim(param, vocab)
    model.train(readdata)
    readdata.stop_and_wait_ansyc()

  if 'predmsg' in tasks:
    param.update(vars(args))
    readdata = ToutiaoSimInput(param)
    vocab = Vocab(param['vocab'], param['emb_size'])
    model = ToutiaoSim(param, vocab)

    if 'predmsg' in tasks and len(args.predset) > 0:
      outfname = os.path.join(args.inputpath, args.predsetoutput)
      with open(outfname, 'w', encoding="utf-8") as outf:
        model.infermsg(readdata, outf)
