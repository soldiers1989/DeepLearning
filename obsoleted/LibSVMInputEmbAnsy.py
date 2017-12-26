# -*- coding: utf-8 -*-

import os
import pickle
import random
import datetime
import threading
import numpy as np
import argparse
from itertools import product
from itertools import islice
from queue import Queue

class TxtFilesConcurrentRandomReaderV2(object):
  # Three files are needed in the path
  def __init__(self, args):
    self.file_batch_size = 15
    self.capacity = 1000
    self.inputpath = args.get('inputpath', '')

    tmpfiles = []
    self.totalfilesize = 0
    filesizemin = 9223372036854775807
    for f in args['dataset']:
      fname = f
      if not os.path.isfile(fname):
        fname = self.inputpath + '/' + f
      if not os.path.isfile(fname):
        print('Input file "' + fname + '" NOT EXISTED')
        continue
      fsize = os.path.getsize(fname)
      self.totalfilesize += fsize
      if fsize < filesizemin: filesizemin = fsize
      tmpfiles.append([fname, fsize])
    [ii.append(round(float(ii[1]) * 1000 / filesizemin)) for ii in tmpfiles]
    self.files = tmpfiles
    print(self.files)

    self.epochs = args.get('epochs', 0)
    self.nowepochs = 0

    self.openfile = []
    self.readfileprobs = []
    self.lines = []
    # self.dataq = PriorityQueue()

    self.nextfile = self._next_file()
    self.nextline = self._next_line()
    self.nextfile_batch = self._next_file_batch()

  def __del__(self):
    self.closeopenfiles()

  def closeopenfiles(self):
    for ii in self.openfile:
      ii[0].close()

  def openfiles(self):
    # [['data//msg_20171031data.bincai', 52030, 1039], ['data//msg_20171101data.bincai', 50073, 1000]]
    self.readfileprobs = []
    sumprob = sum([f[2] for f in self.files])
    for f in self.files:
      fd = open(f[0], 'r')
      self.openfile.append((fd, f[2] * 1.0 / sumprob))
      self.readfileprobs.append(f[2] * 1.0 / sumprob)
    print(self.openfile)
    print(self.readfileprobs)

  def closeopenfile(self, idx):
    self.openfile[idx][0].close()
    del self.openfile[idx]

    sumprob = sum([ii[1] for ii in self.openfile])
    self.readfileprobs = []
    for ii in self.openfile:
      self.readfileprobs.append(ii[1] * 1.0 / sumprob)

  def _next_file(self):
    nextfile_epoch = 0
    self.closeopenfiles()
    self.openfiles()

    while True:
      if len(self.openfile) == 0:
        nextfile_epoch = nextfile_epoch + 1  # 已经读完一轮
        if self.epochs == 0:
          nextfile_epoch = 0
          self.openfiles()
          continue
        else:
          if nextfile_epoch < self.epochs:
            self.openfiles()
            continue
        break

      idx = np.random.choice(len(self.openfile), p=self.readfileprobs)
      yield idx, self.openfile[idx]

  def _next_line(self):
    while True:
      idx, f = next(self.nextfile)
      line = f[0].readline()
      if not line:
        self.closeopenfile(idx)
      else:
        yield line.strip()

  def _next_file_batch(self, size=0):
    if size == 0: size = self.file_batch_size
    while True:
      ret = []
      idx, f = next(self.nextfile)
      for ii in range(size):
        line = f[0].readline()
        if not line:
          self.closeopenfile(idx)
          break
        ret.append(line.strip())

      if len(ret) > 0:
        yield ret

  def read_batch(self, size=256):
    if self.capacity < size:
      self.capacity = size * 3
    while len(self.lines) < self.capacity:
      try:
        newlines = next(self.nextfile_batch)
      except StopIteration:
        break
      self.lines.extend(newlines)
    random.shuffle(self.lines)
    if len(self.lines) < size:
      size = len(self.lines)
    ret = self.lines[:size]
    self.lines = self.lines[size:]
    return ret

class TxtFilesRandomReader(object):
  # Three files are needed in the path
  def __init__(self, dim, files, shuffleFile=True, shuffleRecord=True, epochs=None):
    self.capacity = 1
    self.dim = dim
    self.files = files
    self.shuffleFile = shuffleFile
    self.shuffleRecord = shuffleRecord
    self.epochs = epochs
    self.nextfile = self._get_nextfile()
    self.nextline = self._read_line()
    self.lines = []

  def _get_nextfile(self):
    epoch, retidx = 0, 0

    filelist = list(self.files)
    if len(filelist) == 0:
      print('File list is NULL')
      return
    if self.shuffleFile:
      random.shuffle(filelist)

    while True:
      if retidx >= len(filelist):
        epoch = epoch + 1  # 已经读完一轮
        retidx = 0
        if self.shuffleFile:
          random.shuffle(filelist)
        if self.epochs is None:
          epoch = 0
          continue
        else:
          if epoch < self.epochs:
            continue
        break
      else:
        retidx = retidx + 1
        yield filelist[retidx - 1]

  def get_nextfile(self):
    filename = next(self.nextfile)
    print('Processing ' + filename)
    return filename

  def _read_line(self):
    while True:
      fname = self.get_nextfile()
      with open(fname, 'r') as f:
        for row in f:
          yield row.strip()

  def read_line(self):
    return next(self.nextline)

  def read_batch(self, size=256):
    if self.capacity < size:
      self.capacity = size * 5
    if len(self.lines) < self.capacity:
      self.lines = self.lines + list(islice(self.nextline, self.capacity - len(self.lines)))
    if self.shuffleRecord:
      random.shuffle(self.lines)
    if len(self.lines) < size:
      size = len(self.lines)
    ret = self.lines[:size]
    self.lines = self.lines[size:]
    return ret

# --以下是稀疏的
# 0 agebucket
# 1 genderbucket
# 2 gradebucket
# 3 userfulllongreadldamap,
# 4 userfullshortreadldamap,
# 5 usernotclickldamap,
# 6 msgreadldamap
# 7 userlongreadtagmap,
# 8 usershortreadtagmap,
# 9 usernotclicktagmap,
# 10 msgreadtagmap
# 11 bizuin
# 12 regioncode
# --以下是稠密的
# 13 user_long_readpv
# 14 user_short_readpv
# 15 msgfulllongreadinnerprod
# 16 msgfullshortreadinnerprod
# 17 msglongreadtaginnerprod
# 18 msgshortreadtaginnerprod
# 19 msgnotclickinnerprod
# 20 msgnotclicktaginnerprod
# 21 embed1
# 22 embed2
# [10, 3, 3, 1600, 1600, 1600, 1600, 100000, 100000, 100000, 100000, 10000, 5000, 1, 1, 1, 1, 1, 1, 1, 1, 128, 100]

class LoadLibSvmDataV2(object):
  CONF_INDEX_ORDER = 0
  CONF_INDEX_NAME = 1
  CONF_INDEX_TYPE = 2
  CONF_INDEX_ORG_SIZE = 3
  CONF_INDEX_ORG_BEGIN = 4
  CONF_INDEX_ORG_END = 5
  CONF_INDEX_SIZE = 6
  CONF_INDEX_BEGIN = 7
  CONF_INDEX_END = 8

  def __init__(self, inputargs):
    self.args = inputargs
    self.dataversion = inputargs['data_version']

    if self.dataversion != 7:
      self.process_conf(eval(
'''
[
(0,'agebucket',1,10),
(1,'genderbucket',1,3),
(2,'gradebucket',1,3),
(3,'userfulllongreadldamap',1,1600),
(4,'userfullshortreadldamap',1,1600),
(5,'usernotclickldamap',1,1600),
(6,'msgreadldamap',1,1600),
(7,'userlongreadtagmap',1,300),
(8,'usershortreadtagmap',1,300),
(9,'usernotclicktagmap',1,300),
(10,'msgreadtagmap',1,300),
(11,'bizuin',1,10000),
(12,'regioncode',1,5000),
(13,'level',1,6),
(14,'user_long_readpvbucket',1,10),
(15,'user_short_readpvbucket',1,10),
(16,'msgfulllongreadinnerprod',0,1),
(17,'msgfullshortreadinnerprod',0,1),
(18,'msglongreadtaginnerprod',0,1),
(19,'msgshortreadtaginnerprod',0,1),
(20,'msgnotclickinnerprod',0,1),
(21,'msgnotclicktaginnerprod',0,1),
(22,'titlereadinnerprod',0,1)]
'''  
      ))
    else:
      self.process_conf(eval(
'''
[
(0,'agebucket',1,10),
(1,'genderbucket',1,3),
(2,'gradebucket',1,3),
(3,'userfulllongreadldamap',1,1600),
(4,'userfullshortreadldamap',1,1600),
(5,'usernotclickldamap',1,1600),
(6,'msgreadldamap',1,1600),
(7,'userlongreadtagmap',1,100000),
(8,'usershortreadtagmap',1,100000),
(9,'usernotclicktagmap',1,100000),
(10,'msgreadtagmap',1,100000),
(11,'bizuin',1,10000),
(12,'regioncode',1,5000),
(13,'user_long_readpvbucket',1,10),
(14,'user_short_readpvbucket',1,10),
(15,'msgfulllongreadinnerprodbucket',1,10),
(16,'msgfullshortreadinnerprodbucket',1,10),
(17,'msglongreadtaginnerprodbucket',1,10),
(18,'msgshortreadtaginnerprodbucket',1,10),
(19,'msgnotclickinnerprodbucket',1,10),
(20,'msgnotclicktaginnerprodbucket',1,10),
(21,'user_long_readpv',0,1),
(22,'user_short_readpv',0,1),
(23,'msgfulllongreadinnerprod',0,1),
(24,'msgfullshortreadinnerprod',0,1),
(25,'msglongreadtaginnerprod',0,1),
(26,'msgshortreadtaginnerprod',0,1),
(27,'msgnotclickinnerprod',0,1),
(28,'msgnotclicktaginnerprod',0,1),
(29,'embed',0,100)
]
'''
      ))

    self.statfile = inputargs['statfile']
    self.remove_lowfeq = inputargs['remove_lowfeq']
    self.inputpath = inputargs['inputpath'] + '/'
    if not os.path.isfile(self.statfile):
      self.statfile = self.inputpath + inputargs['statfile']
    self.rmfeature(self.args['remove_feature'])
    self.loadidmap()

    # cross feature
    self.cross_dim = 0
    self.cross_conf = []
    self.cross_raw_feature_map = {}
    self.feature_cross = inputargs['feature_cross']
    self.create_feature_cross_info()

    # embedding feature
    self.embedding_raw_feature_map = {}
    self.embedding_conf = []
    self.create_embedding_lookup_info(inputargs['feature_embedding'])
    self.emb_fnum = len(self.embedding_conf)

    self.dataset = [self.inputpath + item for item in inputargs['dataset']]
    print('self.dataset %s' % self.dataset)
    self.testset = [self.inputpath + item for item in inputargs['testset']]
    print('self.testset %s' % self.testset)
    self.predset = [self.inputpath + item for item in inputargs['predset']]
    print('self.predset %s' % self.predset)

    self.batch_size = inputargs['batch_size']
    if 'batch_size_test' in inputargs:
      self.batch_size_test = inputargs['batch_size_test']
    else:
      self.batch_size_test = self.batch_size * 2

    # self.traindata = TxtFilesRandomReader(dim=self.rawdim, files=self.dataset, shuffleFile=inputargs['shuffle_file'])
    self.traindata = TxtFilesConcurrentRandomReaderV2(inputargs)
    self.testdata = TxtFilesRandomReader(dim=self.rawdim, files=self.testset, shuffleFile=inputargs['shuffle_file'])
    self.preddata = TxtFilesRandomReader(dim=self.rawdim, files=self.predset, shuffleFile=False, shuffleRecord=False,
                                         epochs=1)

    self.dim = self.rawdim + self.cross_dim

    self.ansy_run = True
    self.traindataqueue = Queue(maxsize=5)
    self.testdataqueue = Queue(maxsize=3)
    self.readlock = threading.Lock()
    self.threads = []

  def rmfeature(self, remove_feature):
    self.remove_feature = []
    for item in remove_feature:
      for f in self.feature_conf:
        if item == f[self.CONF_INDEX_NAME]:
          self.remove_feature.append(f)
          break
    print('Remove feature: ' + str(self.remove_feature))

  def process_conf(self, conf):
    begin = 1
    self.feature_conf = []
    for item in conf:
      self.feature_conf.append(item + (begin, begin + item[3] - 1))
      begin += item[3]
    self.feature_num = len(self.feature_conf)

  def loadidmap(self):
    toload = pickle.load(open(self.statfile, "rb"))
    self.idstat = toload['map']
    self.maxidx = toload['maxidx']
    self.idmap = {}
    if 'idmap' in toload:
      self.idmap = toload['idmap']

    vkey = []
    for (k, v) in self.idstat.items():
      if v <= self.remove_lowfeq: continue
      removek = False
      for item in self.remove_feature:
        if item[self.CONF_INDEX_ORG_BEGIN] <= k and k <= item[self.CONF_INDEX_ORG_END]:
          removek = True
          break
      if removek: continue

      vkey.append(k)

    vkey = sorted(vkey)
    self.idmap = dict([(v, k) for (k, v) in list(enumerate(vkey, start=1))])
    self.maxidx = len(vkey)
    self.rawdim = self.maxidx

    begin = 1
    new_feature_conf = []
    for item in self.feature_conf:
      validkey = len(
        [k for k in self.idmap.keys() if item[self.CONF_INDEX_ORG_BEGIN] <= k and k <= item[self.CONF_INDEX_ORG_END]])
      new_feature_conf.append(item + (validkey, begin, begin + validkey - 1))
      begin += validkey

    self.feature_conf = new_feature_conf
    self.printconf()

  def fine_feature_conf_by_name(self, name):
    name = name.strip()
    for item in self.feature_conf:
      if item[self.CONF_INDEX_NAME] == name:
        return item
    return None

  def create_feature_cross_raw(self, idxlist):
    for idx in idxlist:
      item = self.feature_conf[idx]
      offset = 0
      for ii in range(item[self.CONF_INDEX_ORG_BEGIN], item[self.CONF_INDEX_ORG_END] + 1):
        if ii in self.idmap.keys():
          self.cross_raw_feature_map[ii] = (idx, offset)
          offset += 1

  def create_feature_cross2d(self, idx1, idx2):
    item1 = self.feature_conf[idx1]
    item2 = self.feature_conf[idx2]
    self.create_feature_cross_raw([idx1, idx2])
    featuredim = item1[self.CONF_INDEX_SIZE] * item2[self.CONF_INDEX_SIZE]
    self.cross_conf.append((featuredim, 2, (idx1, idx2), (item1, item2)))

  def create_feature_cross3d(self, idx1, idx2, idx3):
    item1 = self.feature_conf[idx1]
    item2 = self.feature_conf[idx2]
    item3 = self.feature_conf[idx3]
    self.create_feature_cross_raw([idx1, idx2, idx3])
    featuredim = item1[self.CONF_INDEX_SIZE] * item2[self.CONF_INDEX_SIZE] * item3[self.CONF_INDEX_SIZE]
    self.cross_conf.append((featuredim, 3, (idx1, idx2, idx3), (item1, item2, item3)))

  def create_feature_cross_info(self):
    self.cross_dim = 0
    self.cross_conf = []
    self.cross_raw_feature_map = {}
    if self.feature_cross:
      self.create_feature_cross2d(0, 1)
      self.create_feature_cross2d(0, 2)
      self.create_feature_cross2d(1, 2)
      self.create_feature_cross3d(0, 1, 2)
      
      self.create_feature_cross2d(0, 11)
      self.create_feature_cross2d(1, 11)
      self.create_feature_cross2d(2, 11)
      self.create_feature_cross3d(0, 1, 11)         
      
      if self.dataversion == 8:
        self.create_feature_cross2d(0, 12)
        self.create_feature_cross2d(2, 12)
        self.create_feature_cross2d(3, 12)
        self.create_feature_cross2d(11, 12)

      new_conf = []
      idx = 0
      begin = self.feature_conf[-1][self.CONF_INDEX_END] + 1
      for item in self.cross_conf:
        new_conf.append((idx, begin, begin + item[0] - 1) + item)
        idx += 1
        begin += item[0]

      self.cross_conf = new_conf

      for item in self.cross_conf:
        self.cross_dim += item[3]
        print(item)

      print('cross_raw_feature_map: ' + str(self.cross_raw_feature_map))

  def create_embedding_lookup_info_item(self, fname):
    item = self.fine_feature_conf_by_name(fname)

    offset = 0
    idx = (len(self.embedding_conf))
    for ii in range(item[self.CONF_INDEX_ORG_BEGIN], item[self.CONF_INDEX_ORG_END] + 1):
      if ii in self.idmap.keys():
        offset += 1
        self.embedding_raw_feature_map[ii] = (idx, offset)
    self.embedding_conf.append((idx, item, offset + 1))

  def create_embedding_lookup_info(self, embedding_feature):
    self.embedding_raw_feature_map = {}
    self.embedding_conf = []
    for ii in list(set(embedding_feature)):
      self.create_embedding_lookup_info_item(ii)

    print('embedding_raw_feature_map: ' + str(self.embedding_raw_feature_map))
    print('embedding_conf: ' + str(self.embedding_conf))
    # {408129: (0, 3), 411139: (0, 6), 414783: (0, 11), 406731: (0, 1), 414828: (0, 12), 412974: (0, 10), 407531: (0, 2), 408304: (0, 4), 411441: (0, 7), 406487: (0, 0), 412346: (0, 9), 409183: (0, 5), 412063: (0, 8)}

  def processing_feature(self, fields):
    feature = np.zeros(self.dim, dtype=float)
    feature_emb = np.zeros(self.emb_fnum, dtype=int) if self.emb_fnum > 0 else None
    cross_raw_feature = {}

    for item in fields[1:]:
      index, value = item.split(':')
      idx = int(index)
      val = float(value)
      if (idx in self.idmap) and (val != 0):
        feature[self.idmap[idx] - 1] = float(value)

        if self.emb_fnum > 0 and (idx in self.embedding_raw_feature_map):
          item = self.embedding_raw_feature_map[idx]
          feature_emb[item[0]] = item[1]

        if self.feature_cross and (idx in self.cross_raw_feature_map):
          item = self.cross_raw_feature_map[idx]
          if item[0] in cross_raw_feature:
            cross_raw_feature[item[0]].append(item)
          else:
            cross_raw_feature[item[0]] = [item, ]

    if self.feature_cross and len(cross_raw_feature) > 1:
      for conf in self.cross_conf:
        for prod in list(product(*[cross_raw_feature.get(ii, []) for ii in conf[5]])):
          offset = 0
          for rawitem, rawconf in zip(prod, conf[6]):
            offset = offset * rawconf[self.CONF_INDEX_SIZE] + rawitem[1]
          feature[conf[1] + offset - 1] = 1.0

    return feature, feature_emb

  def processing_batch(self, lines):
    labels = []
    features = []
    emb = []
    for line in lines:
      fields = line.strip().split()
      label = int(fields[0])
      f, fe = self.processing_feature(fields)
      features.append(f)
      emb.append(fe)
      labels.append(label)

    return {'X': features, 'E': emb, 'Y': labels, 'D': len(lines)}

  def read_traindata_batch(self, size=256):
    with self.readlock:
      lines = self.traindata.read_batch(size)
    return self.processing_batch(lines)

  def do_ansyc_trainset(self):
    print('Begin thread for traindata.read_batch')
    while self.ansy_run:
      with self.readlock:
        now = datetime.datetime.now()
        logingfo = 'do_ansyc_trainset BEGIN: '+now.strftime("%H:%M:%S")
        
        lines = self.traindata.read_batch(self.batch_size)
        
        now2 = datetime.datetime.now()
        logingfo = logingfo+' END READ: '+now2.strftime("%H:%M:%S")
        logingfo = logingfo+' DURA: '+str(now2-now)
      self.traindataqueue.put(self.processing_batch(lines))
      
      now3 = datetime.datetime.now()
      logingfo = logingfo+' END PROC: '+now3.strftime("%H:%M:%S")
      logingfo = logingfo+' DURA: '+str(now3-now2)+' SIZE: '+str(self.batch_size)
      print(logingfo)
    print('End thread for traindata.read_batch')

  def do_ansyc_testset(self):
    print('Begin thread for traindata.read_batch')
    if self.has_testset():
      while self.ansy_run:        
        now = datetime.datetime.now()
        logingfo = 'do_ansyc_testset BEGIN: '+now.strftime("%H:%M:%S")
      
        with self.readlock:
          lines = self.testdata.read_batch(self.batch_size_test)
          
          now2 = datetime.datetime.now()
          logingfo = logingfo+' END READ: '+now2.strftime("%H:%M:%S")
          logingfo = logingfo+' DURA: '+str(now2-now)
        self.testdataqueue.put(self.processing_batch(lines))
        
        now3 = datetime.datetime.now()
        logingfo = logingfo+' END PROC: '+now3.strftime("%H:%M:%S")
        logingfo = logingfo+' DURA: '+str(now3-now2)+' SIZE: '+str(self.batch_size_test)
        print(logingfo)
    print('End thread for traindata.read_batch')

  def start_ansyc(self):
    self.threads = []
    for ii in range(3):
      t = threading.Thread(target=self.do_ansyc_trainset)
      self.threads.append(t)
      t.start()
    t = threading.Thread(target=self.do_ansyc_testset)
    self.threads.append(t)
    t.start()

  def stop_and_wait_ansyc(self):
    self.ansy_run = False
    for t in self.threads:
      while not self.traindataqueue.empty():
        self.traindataqueue.get()
      while not self.testdataqueue.empty():
        self.testdataqueue.get()
      t.join()

  def read_traindata_batch_ansyc(self):
    return self.traindataqueue.get()

  def has_testset(self):
    return len(self.testset) > 0

  def read_testdata_batch(self, size=256):    
    with self.readlock:
      lines = self.testdata.read_batch(size)
    return self.processing_batch(lines)

  def read_testdata_batch_ansyc(self):
    return self.testdataqueue.get()

  def read_preddata_batch(self, size=256):
    labels = []
    features = []
    emb = []
    lines = self.preddata.read_batch(size)
    for line in lines:
      fields = line.strip().split()
      f, fe = self.processing_feature(fields)
      features.append(f)
      emb.append(fe)
      labels.append(fields[0])

    return {'X': features, 'E': emb, 'ID': labels, 'D': len(lines)}

  def has_predset(self):
    return len(self.predset) > 0

  def reset_predset(self):
    self.preddata = TxtFilesRandomReader(dim=self.rawdim, files=self.predset, shuffleFile=False, shuffleRecord=False,
                                         epochs=1)

  def printconf(self):
    for item in self.feature_conf:
      print(item)
    print('load ' + self.statfile + ', maxidx: ' + str(self.maxidx))

  def printinfoanddata(self):
    self.start_ansyc()
    print(self.read_traindata_batch_ansyc())
    print(self.read_traindata_batch_ansyc())
    print(self.read_traindata_batch_ansyc())
    self.stop_and_wait_ansyc()

  #    for f in self.dataset+self.testset:
  #      print('Processing file:'+f)
  #      reader=TxtFilesRandomReader(dim=self.dim, files=[f], shuffleFile=self.args['shuffle_file'])
  #      self.print_data(reader.read_line())
  #    print(self.read_traindata_batch(size=10))
  #    print(self.read_testdata_batch(size=10))
  #    if self.has_predset():
  #      print(self.read_preddata_batch(size=3))
  #    else:
  #      print('Skip pred file')

  def parse_data(self, line):
    fields = line.strip().split()
    label = int(fields[0])

    feaures = [(item.split(':')) for item in fields[1:]]
    feaures = sorted(filter(lambda x: x[1] != 0, [(int(item[0]), float(item[1])) for item in feaures]),
                     key=lambda a: a[0])
    return label, feaures

  def print_data(self, line):
    print(line)
    _, feaures = self.parse_data(line)

    fidx = 0
    for item in feaures:
      while fidx < self.feature_num:
        if self.feature_conf[fidx][self.CONF_INDEX_ORG_BEGIN] <= item[0] and item[0] <= self.feature_conf[fidx][
          self.CONF_INDEX_ORG_END]:
          if item[0] not in self.idmap:
            print(" Name: " + self.feature_conf[fidx][self.CONF_INDEX_NAME] + " Value:" + str(item) + " skipped")
          else:
            print(
              " Name: " + self.feature_conf[fidx][self.CONF_INDEX_NAME] + " Value:" + str(item) + " Now index:" + str(
                self.idmap[item[0]]))
          break
        else:
          fidx += 1
      if fidx > self.feature_num:
        print(" Unknown: " + self.feature_conf[fidx][self.CONF_INDEX_NAME] + " Value:" + str(item))

    if self.has_predset():
      ret = self.read_preddata_batch(size=3)
      while ret['D'] > 0:
        print(ret)
        ret = self.read_preddata_batch(size=3)
      self.reset_predset()
      ret = self.read_preddata_batch(size=5)
      while ret['D'] > 0:
        print(ret)
        ret = self.read_preddata_batch(size=5)


def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run Reindex.")
  parser.add_argument('--data_version', type=int, default=6,
                      help='Data version')
  parser.add_argument('--inputpath', default='data/', required=False,
                      help='Input data path.')
  parser.add_argument('--dataset', nargs='+', default=['msg_20171031data.bincai', 'msg_20171101data.bincai'],
                      help='Choose a train dataset.')
  parser.add_argument('--testset', nargs='*', default=[],
                      help='Choose a test dataset.')
  parser.add_argument('--predset', nargs='?', default='',
                      help='Choose a pred dataset.')
  parser.add_argument('--shuffle_file', type=str2bool, default=True,
                      help='Suffle input file')
  parser.add_argument('--batch_size', type=int, default=10,
                      help='Data batch size')
  parser.add_argument('--batch_size_test', type=int, default=0,
                      help='Data batch size')
  parser.add_argument('--remove_feature', nargs='*', default=[], required=False,
                      help='Remove Some Feature')
  parser.add_argument('--remove_lowfeq', type=int, default=0,
                      help='Remove Low Frequence Feature (include)')
  parser.add_argument('--statfile', default='statfile', required=False,
                      help='stat mapping file.')
  parser.add_argument('--feature_cross', type=str2bool, default=False,
                      help='cross feature.')
  parser.add_argument('--feature_embedding', nargs='*', default=[], required=False,
                      help='embedding feature.')
  parser.add_argument('--printinfoanddata', type=str2bool, default=True,
                      help='print stat info.')
  args = parser.parse_args()
  print(vars(args))
  readdata = LoadLibSvmDataV2(vars(args))
  if args.printinfoanddata:
    readdata.printinfoanddata()

# tdata=['0 9:1 13:1 16:1 107:0.0109 115:0.0639 134:0.0270 262:0.0103 289:0.0193 453:0.0133 526:0.0136 533:0.0171 561:0.0177 565:0.0118 590:0.0447 637:0.0154 658:0.0222 712:0.0194 722:0.0174 772:0.0276 794:0.0122 985:0.0151 1013:0.0450 1114:0.0310 1126:0.0266 1262:0.0200 1296:0.0119 1316:0.0102 1446:0.0317 1492:0.0422 1506:0.0105 1634:0.0137 1641:0.0210 1734:0.0227 1808:0.0109 1903:0.0111 1917:0.0237 2138:0.0106 2139:0.0142 2144:0.0114 2190:0.0327 2258:0.0124 2355:0.0234 2455:0.0116 2479:0.0464 2494:0.0173 2498:0.0167 2613:0.0848 2714:0.0164 2731:0.0150 2736:0.0177 2748:0.0179 2952:0.0158 3092:0.0143 3124:0.0207 3133:0.0229 3232:0.0470 3234:0.1189 3350:0.0124 3403:0.0151 3421:0.0167 3518:0.0189 3531:0.0148 3578:0.0105 3601:0.0151 3636:0.0149 3691:0.0334 3887:0.0155 3948:0.0138 3955:0.0108 3990:0.0578 4098:0.1322 4335:0.0322 4348:0.1253 4645:0.0173 4683:0.0825 4716:0.0229 5867:0.0896 5948:0.6628 6084:0.0420 407798:1 418128:1 421417:0.1430 421418:0.1009 421419:0.0000 421420:0.1252 421421:0.0000 421422:0.0000 421423:0.4816 421553:0.0088 421554:0.0225 421555:0.1119 421556:-0.1752 421557:-0.1107 421558:0.0432 421559:0.0237 421560:-0.1678 421561:0.0003 421562:0.0308 421563:-0.0935 421564:-0.0022 421565:-0.0529 421566:-0.0007 421567:-0.0311 421568:-0.0456 421569:0.1593 421570:-0.01 421571:0.0199 421572:0.0625 421573:-0.0461 421574:0.0283 421575:0.0014 421576:-0.0659 421577:-0.0676 421578:-0.2543 421579:-0.0065 421580:-0.2427 421581:0.0241 421582:-0.0114 421583:-0.0497 421584:0.0359 421585:0.0734 421586:-0.0051 421587:0.025 421588:-0.0289 421589:0.0235 421590:-0.1095 421591:0.1251 421592:0.03 421593:0.1832 421594:-0.1141 421595:0.0527 421596:0.0664 421597:0.1162 421598:0.0707 421599:-0.0168 421600:-0.1876 421601:-0.0206 421602:0.0479 421603:-0.1697 421604:0.0143 421605:-0.0078 421606:0.0376 421607:0.0216 421608:-0.1257 421609:0.0468 421610:-0.0078 421611:-0.025 421612:0.054 421613:-0.0104 421614:-0.0605 421615:0.0011 421616:-0.1121 421617:-0.2165 421618:0.0522 421619:-0.0234 421620:0.02 421621:-0.0455 421622:-0.0363 421623:-0.1341 421624:0.0753 421625:-0.0295 421626:-0.1534 421627:-0.0075 421628:0.2945 421629:0.0468 421630:-0.5243 421631:0.2086 421632:0.3993 421633:0.0217 421634:-0.1039 421635:0.0056 421636:0.0072 421637:0.148 421638:-0.0585 421639:0.0159 421640:-0.2327 421641:0.0444 421642:-0.0581 421643:-0.0204 421644:0.0228 421645:-0.0229 421646:-0.0674 421647:0.0335 421648:-0.0686 421649:-0.0953 421650:-0.1477 421651:0.0506 421652:-0.0875']
#
#  print(readdata.processing_batch(tdata))

