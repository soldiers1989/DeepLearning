# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import threading
import numpy as np
from itertools import islice


class TxtFileReader(object):
  def __init__(self, file):
    self.DEFAULT_BATCH_SIZE = 10240
    self.file = file
    self.nextline = self._read_line()

  def _read_line(self):
    with open(self.file, 'r') as rd:
      while True:
        line = rd.readline()
        if not line: break
        yield line.strip()

  def read_line(self):
    return next(self.nextline)

  def read_batch(self, size=0):
    if size == 0: size = self.DEFAULT_BATCH_SIZE
    return list(islice(self.nextline, size))

  def read_sparse_batch(self, size=0):
    if size == 0: size = self.DEFAULT_BATCH_SIZE
    labels = []
    features = []
    cnt = 0
    for line in self.read_batch(size):
      cnt += 1
      if '#' in line:
        punc_idx = line.index('#')
      else:
        punc_idx = len(line)
      label = float(line[0:1])
      if label > 1:
        label = 1
      feature_line = line[2:punc_idx]
      words = feature_line.split(' ')
      cur_feature_list = []
      for word in words:
        if not word:
          continue
        tokens = word.split(':')

        value = 0.0
        if len(tokens[1]) > 0:
          value = float(tokens[1])
        if value != 0.0:
          cur_feature_list.append([int(tokens[0]), value])

      features.append(cur_feature_list)
      labels.append(label)

    return cnt, labels, features
    
  def stat_batch(self, size=0):
    if size == 0: size = self.DEFAULT_BATCH_SIZE
    idstat = {}
    cnt = 0
    totalfield = 0
    for line in self.read_batch(size):
      cnt += 1
      if '#' in line:
        punc_idx = line.index('#')
      else:
        punc_idx = len(line)
      feature_line = line[2:punc_idx]
      words = feature_line.split(' ')
      for word in words:
        if not word:
          continue
        tokens = word.split(':')
        value = 0.0
        if len(tokens[1]) > 0:
          value = float(tokens[1])
        if value != 0.0:
          totalfield += 1
          key=int(tokens[0])
          if key not in idstat:
            idstat[key] = 1
          else:
            idstat[key] += 1

    return cnt, totalfield, idstat


class StatData(object):
  def __init__(self, inputargs):
    self.args = inputargs
    self.inputpath = args.inputpath  # 输入目录
    self.outputpath = args.outputpath  # 输出目录
    self.inputfiles = args.inputfiles  # 待处理文件列表

    self.idmaplock = threading.Lock()
    self.idstat = {}
    self.idmap = {}
    self.totalline = 0
    self.totalfield = 0
    self.maxidx = 0
    self.statfile = self.outputpath + '/' + inputargs.statfile

  def filestat(self, fname):
    tfr = TxtFileReader(fname)
    cnt, totalfield, fidstat = tfr.stat_batch()
    while cnt > 0:
      with self.idmaplock:
        self.totalline += cnt
        self.totalfield += totalfield
        for k in fidstat:
          if k not in self.idstat:
            self.idstat[k] = fidstat[k]
          else:
            self.idstat[k] += fidstat[k]
      cnt, totalfield, fidstat = tfr.stat_batch()

  def statdata(self):
    if not os.path.exists(self.inputpath):
      print('Input dir "' + self.inputpath + '" NOT EXISTED')
      exit
    files = self.inputfiles.split('#')
    threads = []
    for f in files:
      fname = self.inputpath + '/' + f
      if not os.path.isfile(fname):
        print('Input file "' + fname + '" NOT EXISTED')
        continue
      print("processing:" + fname)  # self.statfile(fname)
      t = threading.Thread(target=self.filestat, args=(fname,))
      threads.append(t)
      t.start()

    for t in threads:
      t.join()

    self.saveidmap()
    print('Total line ' + str(self.totalline) + ', field: ' + str(self.totalfield))

  def reidxrecord(self, flist):
    it = iter(flist)
    while True:
      try:
        idx, value = next(it)
        if value == 0: continue
        if idx not in self.idmap: continue
        yield (self.idmap[idx], value)
      except StopIteration:
        break  # 遇到StopIteration就退出循环

  def reindexfile(self, fname):
    outfname = self.outputpath + '/' + fname + '.' + self.args.postfix
    fname = self.inputpath + '/' + fname
    tfr = TxtFileReader(fname)
    with open(outfname, 'w') as outf:
      cnt, labels, features = tfr.read_sparse_batch()
      while cnt > 0:
        for label, feature in zip(labels, features):
          reidx_feature_list = sorted(self.reidxrecord(feature), key=lambda a: a[0], reverse=False)
          if len(reidx_feature_list) > 0:
            reidx_feature_list = ' '.join([str(item[0]) + ':' + str(item[1]) for item in reidx_feature_list])
            outf.write('1 ' if label > 0 else '0 ')
            outf.write(reidx_feature_list + '\n')
        cnt, labels, features = tfr.read_sparse_batch()

  def reindexdata(self):
    if not os.path.exists(self.inputpath):
      print('Input dir "' + self.inputpath + '" NOT EXISTED')
      exit

    if not os.path.isfile(self.statfile):
      print('Input file "' + self.statfile + '" NOT EXISTED')
      exit

    self.loadidmap()

    okfield = sorted([k for (k, v) in self.idstat.items()
                      if (v > self.args.remove_lowfeq) and (k < 406552 or not self.args.remove_embedding)])
    #    for (k, v) in self.idstat.items():
    #       not ((v<=self.args.remove_lowfeq) or (k>=406552 and self.args.remove_embedding))
    #           (not v<=self.args.remove_lowfeq) and (not (k>=406552 and self.args.remove_embedding) )
    #           ( v>self.args.remove_lowfeq) and ( k<406552 or not self.args.remove_embedding)
    #      if v<=self.args.remove_lowfeq: continue
    #      if k>=406552 and self.args.remove_embedding: continue
    #      self.maxidx+=1
    #      self.idmap[k]=self.maxidx

    self.idmap = dict([(v, k) for (k, v) in list(enumerate(okfield, start=1))])
    self.maxidx = len(okfield)
    print('self.maxidx: %d' % self.maxidx)

    files = self.inputfiles.split('#')
    threads = []
    for f in files:
      fname = self.inputpath + '/' + f
      if not os.path.isfile(fname):
        print('Input file "' + fname + '" NOT EXISTED')
        continue
      print("processing:" + fname)  # self.statfile(fname)
      t = threading.Thread(target=self.reindexfile, args=(f,))
      threads.append(t)
      t.start()

    for t in threads:
      t.join()

    self.saveidmap()
    print('Total line ' + str(self.totalline) + ', field: ' + str(self.totalfield))

  def saveidmap(self):
    tosave = {"map": self.idstat, "maxidx": self.maxidx, "idmap": self.idmap}
    print('save ' + self.statfile + ', maxidx: ' + str(self.maxidx))
    pickle.dump(tosave, open(self.statfile, 'wb'))
    stat = np.zeros(12)
    for (k, v) in self.idstat.items():
      if v > 100:
        v = 11
      else:
        if v > 10: v = 10
      stat[v] += 1
    print(stat)

  def loadidmap(self):
    toload = pickle.load(open(self.statfile, "rb"))
    self.idstat = toload['map']
    self.maxidx = toload['maxidx']
    self.idmap = {}
    if 'idmap' in toload:
      self.idmap = toload['idmap']
    print('load ' + self.statfile + ', maxidx: ' + str(self.maxidx))
    
  def mergedata(self):    
    if not os.path.exists(self.inputpath):
      print('Input dir "' + self.inputpath + '" NOT EXISTED')
      exit
      
    for f in self.inputfiles:
      fname = self.inputpath + '/' + f
      if not os.path.isfile(fname):
        print('Input file "' + fname + '" NOT EXISTED')
        continue
        
      toload = pickle.load(open(fname, "rb"))
      idstat = toload['map']
      maxidx = toload['maxidx']
      print('load %s, count: %d, maxidx: %d' % ( fname, len(idstat), self.maxidx) )
        
      for kk in idstat: 
        if kk not in self.idstat:
          self.idstat[kk] = idstat[kk]
        else:
          self.idstat[kk] += idstat[kk]
        
        if kk>self.maxidx: self.maxidx=kk

    self.saveidmap()

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
  cmdinfo = '\n  statdata: stat files'
  cmdinfo += '\n  mergedata: merge data'
  parser = argparse.ArgumentParser(usage='%(prog)s [options]' + cmdinfo)
  subparsers = parser.add_subparsers()

  # 统计数据
  stat_data = subparsers.add_parser('statdata')
  stat_data.add_argument('--inputpath', default='data/', required=False,
                         help='Input data path.')
  stat_data.add_argument('--inputfiles', nargs='?', default='', required=True,
                         help='Choose inputfiles, seperated by #.')
  stat_data.add_argument('--outputpath', default='data/', required=False,
                         help='Output data path.')
  stat_data.add_argument('--statfile', default='statfile', required=False,
                         help='stat mapping file.')
  stat_data.set_defaults(func=lambda args: StatData(args).statdata())

  stat_data = subparsers.add_parser('mergedata')
  stat_data.add_argument('--inputpath', default='data/', required=False,
                         help='Input data path.')
  stat_data.add_argument('--inputfiles', nargs='+', default=[], required=True,
                         help='Choose inputfiles, seperated by #.')
  stat_data.add_argument('--outputpath', default='data/', required=False,
                         help='Output data path.')
  stat_data.add_argument('--statfile', default='mergedata', required=False,
                         help='stat mapping file.')
  stat_data.set_defaults(func=lambda args: StatData(args).mergedata())

  args = parser.parse_args()
  print('args: ' + str(args))
  args.func(args)


#  # 重写下标
#  reindex_data = subparsers.add_parser('reindexdata')
#  reindex_data.add_argument('--inputpath', default='data/', required=False,
#                            help='Input data path.')
#  reindex_data.add_argument('--inputfiles', nargs='?', default='', required=True,
#                            help='Choose inputfiles, seperated by #.')
#  reindex_data.add_argument('--outputpath', default='data/', required=False,
#                            help='Output data path.')
#  reindex_data.add_argument('--postfix', default='ridx',
#                            help='Reindex postfix.')
#  reindex_data.add_argument('--statfile', default='statfile', required=False,
#                            help='stat mapping file.')
#  reindex_data.add_argument('--remove_embedding', type=str2bool, default=False,
#                            help='Remove Embedding Feature')
#  reindex_data.add_argument('--remove_lowfeq', type=int, default=0,
#                            help='Remove Low Frequence Feature (include)')
#  reindex_data.set_defaults(func=lambda args: StatData(args).reindexdata())

#args: Namespace(func=<function <lambda> at 0x0000000003060D90>, inputfiles='msg_20171031data.bincai#msg_20171101data.bincai#msg_20171102data.bincai', inputpath='data/', outputpath='data/', statfile='statfile')
#processing:data//msg_20171031data.bincai
#processing:data//msg_20171101data.bincai
#processing:data//msg_20171102data.bincai
#save data//statfile, maxidx: 0
#[   0.  743.  192.  122.  660.   81.   55.   32.  132.   44.  320.    0.]
#Total line 60, field: 16851


