#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse

import numpy as np
from TFBCUtils import Vocab


def loadvedio(args):
  vedioemb = np.zeros( (args.vedio_size, args.emb_size), dtype = np.float32 ) + 10
  id2info = {}
  with open(args.predvediooutput, 'r', encoding="utf-8") as inf:
    for line in inf.readlines():
      items=line.strip().split(' ')
      data = [float(item) for item in items[-100:]]
      itemsitems=items[0].strip().split('|')
      idx=int(itemsitems[1])
      vedioemb[idx-1]=data
      id2info[idx-1]=items[0]
  failed=0
  for ii in range(args.vedio_size):
    if vedioemb[ii][0]>2 : failed=1
  return failed, vedioemb, id2info

def next_user(args):
  previousline=None
  previousuin=-1
  maxidx=-1
  retitems={}
  #1000023805|11|7845|a063268m47w|小狐狸幻化成人的过程！|0.0064978236 -0.018127618 -0.02130943 0.16288851 -0.009042299 -0.0113481935 -0.012320594 0.014147561 -0.029003108 0.18691552 0.076581635 -0.037571993 -0.021198519 0.012083076 0.32306546 -0.03304494 0.3237665 -0.0006660345 -0.0060104937 -0.06337515 0.027764505 -0.02096276 -0.03957079 -0.009013531 -0.029037746 -0.025875594 0.012184887 -0.0022252856 -0.0024346213 0.13695227 -0.005069357 -0.037644465 0.050372493 0.0077712527 -0.028085334 -0.0743781 -0.039322734 -0.029765816 -0.005142185 -0.011552918 -0.024931919 -0.032223366 -0.01658538 -0.024567137 0.32300955 -0.012782609 -0.010093141 0.32288346 0.01105587 -0.044725984 -0.016769148 0.3227682 0.0011340706 -0.040108413 0.0035904385 0.0056958245 0.081572756 -0.020395089 0.044622134 -0.039317448 -0.021131197 -0.037337665 0.2954914 -0.0109236855 3.520723e-05 0.024343576 -0.0051691174 -0.0038317868 0.32235408 -0.021227384 -0.0031964453 -0.004754274 -0.04579926 0.0073204394 -0.00022966502 -0.022512445 -0.0056434874 4.7469643e-05 -0.03420956 -0.040864147 0.030558063 -0.045880392 -0.015945341 -0.021154353 0.18890682 0.015073399 0.0104318205 0.0059514693 0.008678894 0.008796115 -0.00082164747 -0.019746922 0.3225042 -0.0061552357 0.02925103 -0.0026290105 -0.030528262 -0.0001434253 -0.034335118 -0.02036889

  with open(args.preduseroutput, 'r', encoding="utf-8") as inf:
    for line in inf.readlines():
      items=line.strip().split('|')
      if len(items)!=6 : continue
      uin=int(items[0]); od=int(items[1]); nvid=int(items[2])
      vec=[float(ii) for ii in items[5].strip().split(' ')]
      name=items[3]+'|'+items[4]
      if previousuin<0: previousuin=uin
      
      if uin!=previousuin:
        retuin=previousuin
        previousuin=uin
        if len(retitems)==maxidx: #数据OK
          ret=[retitems[ii+1] for ii in range(maxidx)]
          retitems={od: (od, nvid, name, vec)}
          maxidx = od
          yield retuin, ret
        else: #数据不OK
          retitems={od: (od, nvid, name, vec)}
      else:
        if od > maxidx: maxidx = od
        retitems[od]=(od, nvid, name, vec)

def printMetrics(totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400):
  print('total item: %d'%totalcnt)
  print('avg nrr: %f'%(nrr/totalcnt))
  print('avg recall10: %f'%(1.0*recall10/totalcnt))
  print('avg recall20: %f'%(1.0*recall20/totalcnt))
  print('avg recall40: %f'%(1.0*recall40/totalcnt))
  print('avg recall100: %f'%(1.0*recall100/totalcnt))
  print('avg recall200: %f'%(1.0*recall200/totalcnt))
  print('avg recall400: %f'%(1.0*recall400/totalcnt))
    
def computeMetrics(itemindex, totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400):
  totalcnt+=1
  nrr+=1.0/itemindex
  if itemindex<=10: recall10+=1
  if itemindex<=20: recall20+=1
  if itemindex<=40: recall40+=1
  if itemindex<=100: recall100+=1
  if itemindex<=200: recall200+=1
  if itemindex<=400: recall400+=1  
         
  if totalcnt%20000==0 and totalcnt>0: printMetrics(totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400)
  
  return totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400

def evalpred(args):
  failed, vedioemb, id2info = loadvedio(args)
  if failed:
    print('Load embedding error')
    return
  vedioembT=vedioemb.T
  
  totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400=0, 0.0, 0, 0, 0, 0, 0, 0
    
  for uin, items in next_user(args):
    for ii in range(len(items)-1):
      vec=np.array(items[ii][3])
      pred=np.dot(vec, vedioembT)
      argpred=np.argsort(-pred)+1 #视频下标从1开始
      itemindex = np.argwhere(argpred == items[ii+1][1])[0][0] + 1  #视频下标从1开始
      computeMetrics(itemindex, totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400)
        
  printMetrics(totalcnt, nrr, recall10, recall20, recall40, recall100, recall200, recall400)

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
  parser = argparse.ArgumentParser(description="Run Vedio Classify NN.")
  parser.add_argument('--func', default='evalpred',
                      help='Select function printvocab evalpred.')
  parser.add_argument('--vocab', nargs='?', default='data/model2.vec.proc',
                      help='Vocal to load.')                      
  parser.add_argument('--emb_size', type=int, default=100,
                      help='Vocal embedding size.')
  parser.add_argument('--vocab_list', nargs='?', default='1460 4713 33669 18182 1180 238 19023 4713',
                      help='Words to print.')
                      
  parser.add_argument('--predvediooutput', nargs='?', default='data/vedio20000.pred',
                      help='Choose a pred dataset.')
  parser.add_argument('--preduseroutput', nargs='?', default='data/user.pred.sort.2000',
                      help='Choose a pred dataset.')               
  parser.add_argument('--vedio_size', type=int, default=20000,
                      help='Vedio size.')

  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  
  if args.func=='pvocab':
    vocab = Vocab(args.vocab, args.emb_size)
    vocab_list = [int(ii) for ii in args.vocab_list.strip().split(' ')]
    print(vocab_list)
    print(str(vocab.id2string(vocab_list)))
    
  if args.func=='evalpred':
    evalpred(args)
    
    

