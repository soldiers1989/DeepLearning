#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import datetime
import os
import random
import sys
import threading
from itertools import islice

import numpy as np
from TFBCUtils import Vocab

Py3 = sys.version_info[0] == 3
if Py3:
  from queue import Queue
else:
  from Queue import Queue

class TxtFilesConcurrentRandomReaderV2(object):
  # Three files are needed in the path
  def __init__(self, args):
    self.file_batch_size = 15
    self.capacity = 1000
    self.inputpath = args.get('inputpath', '')

    tmpfiles = []
    self.totalfilesize = 0
    filesizemin = 9223372036854775807
    for f in args.get('dataset',[]):
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
    print('TxtFilesConcurrentRandomReaderV2:'+str(self.files))

    self.epochs = args.get('inputepochs', 0)
    self.nowepochs = 0

    self.openfile = []
    self.readfileprobs = []
    self.lines = []
    # self.dataq = PriorityQueue()

    self.nextfile = self._next_file()
    self.nextline = self._next_line()
    self.nextfile_batch = self._next_file_batch()
    self.verbose = 0

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
      #fd = open(f[0], 'r')
      fd = codecs.open(f[0], 'r', 'utf-8')
      self.openfile.append((fd, f[2] * 1.0 / sumprob))
      self.readfileprobs.append(f[2] * 1.0 / sumprob)
    if self.verbose==2: print(self.openfile)
    if self.verbose==2: print(self.readfileprobs)

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
  def __init__(self, files, shuffleFile=True, shuffleRecord=True, epochs=None):
    self.capacity = 1
    self.files = files
    self.shuffleFile = shuffleFile
    self.shuffleRecord = shuffleRecord
    self.epochs = epochs
    self.nextfile = self._get_nextfile()
    self.nextline = self._read_line()
    self.lines = []
    self.verbose = 0

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
    #print('Processing ' + filename)
    return filename

  def _read_line(self):
    while True:
      fname = self.get_nextfile()
      if Py3:
        with open(fname, 'r', encoding='utf-8') as f:
          for row in f:
            yield row.strip()
      else:
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

class VedioClassifyBizuinInputAnsyEmb(object):
  def __init__(self, inputargs):
    self.args = inputargs    
    print('VedioClassifyBizuinInputAnsyEmb:', str(self.args))
    self.inputpath = inputargs['inputpath']
    if not self.inputpath.endswith(''):
      self.inputpath+=os.sep

    self.dataset = []
    self.testset = []
    self.predset = []
    
    if Py3:
      if 'dataset' in inputargs:
        self.dataset = [self.inputpath + item for item in inputargs['dataset']]
        print('self.dataset %s' % self.dataset)
      if 'testset' in inputargs:
        self.testset = [self.inputpath + item for item in inputargs['testset']]
        print('self.testset %s' % self.testset)
      if 'predset' in inputargs:
        self.predset = [self.inputpath + item for item in inputargs['predset']]
        print('self.predset %s' % self.predset)

    else:
      if inputargs.has_key('dataset'):
        self.dataset = [self.inputpath + item for item in inputargs['dataset']]
        print('self.dataset %s' % self.dataset)
      if inputargs.has_key('testset'):
        self.testset = [self.inputpath + item for item in inputargs['testset']]
        print('self.testset %s' % self.testset)
      if inputargs.has_key('predset'):
        self.predset = [self.inputpath + item for item in inputargs['predset']]
        print('self.predset %s' % self.predset)

    self.batch_size = inputargs.get('batch_size', 0)
    self.batch_size_test = inputargs.get('batch_size_test', self.batch_size * 2)

    self.vocab = Vocab(inputargs['vocab'], inputargs['emb_size'], inputargs['vocab_size'])
    self.titlemaxsize = 20
    self.articlemaxsize = 200
    
    self.traindata = TxtFilesConcurrentRandomReaderV2(inputargs)

    self.testdata = TxtFilesRandomReader(files=self.testset, shuffleFile=inputargs.get('shuffle_file', False) )

    self.preddata = TxtFilesRandomReader(files=self.predset, shuffleFile=False, shuffleRecord=False,
                                         epochs=1)

    self.ansy_run = True
    self.traindataqueue = Queue(maxsize=5)
    self.testdataqueue = Queue(maxsize=3)
    self.readlock = threading.Lock()
    self.threads = []
    self.verbose = 0

    self.level1_size=28
    self.level2_size=174

  def read_traindata_batch(self, size=32):
    with self.readlock:
      lines = self.traindata.read_batch(size)
    return self.processing_batch(lines)

  def do_ansyc_trainset(self):
    print('Begin thread for traindata.read_batch')
    while self.ansy_run:
      now = datetime.datetime.now()
      logingfo = 'do_ansyc_trainset BEGIN: '+now.strftime("%Y-%m-%d %H:%M:%S")

      with self.readlock:
        lines = self.traindata.read_batch(self.batch_size)

        now2 = datetime.datetime.now()
        logingfo = logingfo+' END READ: '+now2.strftime("%Y-%m-%d %H:%M:%S")
        logingfo = logingfo+' DURA: '+str(now2-now)
      self.traindataqueue.put(self.processing_batch(lines))

      now3 = datetime.datetime.now()
      logingfo = logingfo+' END PROC: '+now3.strftime("%Y-%m-%d %H:%M:%S")
      logingfo = logingfo+' DURA: '+str(now3-now2)+' SIZE: '+str(self.batch_size)
      if self.verbose==2: print(logingfo)
    print('End thread for traindata.read_batch')

  def do_ansyc_testset(self):
    print('Begin thread for testdata.read_batch')
    if self.has_testset():
      while self.ansy_run:
        now = datetime.datetime.now()
        logingfo = 'do_ansyc_testset BEGIN: '+now.strftime("%Y-%m-%d %H:%M:%S")

        with self.readlock:
          lines = self.testdata.read_batch(self.batch_size_test)

          now2 = datetime.datetime.now()
          logingfo = logingfo+' END READ: '+now2.strftime("%Y-%m-%d %H:%M:%S")
          logingfo = logingfo+' DURA: '+str(now2-now)
        self.testdataqueue.put(self.processing_batch(lines))

        now3 = datetime.datetime.now()
        logingfo = logingfo+' END PROC: '+now3.strftime("%Y-%m-%d %H:%M:%S")
        logingfo = logingfo+' DURA: '+str(now3-now2)+' SIZE: '+str(self.batch_size_test)
      if self.verbose==2: print(logingfo)
    print('End thread for testdata.read_batch')

  def start_ansyc(self):
    self.threads = []
    if len(self.dataset)>0:
      for ii in range(3):
        t = threading.Thread(target=self.do_ansyc_trainset)
        self.threads.append(t)
        t.start()
    else:
      print('*'*20)
      print('self.dataset is NULL')
      print('*'*20)

    if len(self.dataset)>0:
      t = threading.Thread(target=self.do_ansyc_testset)
      self.threads.append(t)
      t.start()
    else:
      print('*'*20)
      print('self.testset is NULL')
      print('*'*20)

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

  def has_predset(self):
    return len(self.predset) > 0

  def reset_predset(self):
    self.preddata = TxtFilesRandomReader(files=self.predset, shuffleFile=False, shuffleRecord=False,
                                         epochs=1)

  def parseBizClass(self, line, biznum, label):
    ret = np.zeros(biznum)
    if line=='NULL' or line=='': return ret

    data = [item.strip().split(':') for item in line.strip().split('#')]
    for item in data:
      key=int(item[0])
      ret[key] = float(item[1])-1 if key==int(label) else float(item[1])
    norm = sum(ret)
    if norm>5: ret=ret/norm
    else: ret = np.zeros(biznum)
    return ret
    
  def parseLDAArray(self, line, topicnum):
    data = [item.strip().split(':') for item in line.strip().split('#')]
    norm = sum([float(item[1]) for item in data])
    ret = np.zeros(topicnum)
    for item in data:
      ret[int(item[0])]=float(item[1]) /norm
    return ret
    
  def parseVocab(self, line, maxsize):    
    ret = self.vocab.string2id([item for item in line.strip().split(' ')])
    linelen=len(ret)
    if(linelen<maxsize): ret.extend( [0]*(maxsize-len(ret)) )
    return np.array(ret[:maxsize]), min(linelen, maxsize)
    
  def parseLable(self, line, labelnum):
    ret = np.zeros(labelnum)
    ret[int(line)] = 1.0
    return ret

#SELECT concat( regexp_replace(concat(vid, '_', bizuin, '_', mid, '_', idx, '_', title, '_', vtitle, '_', nickname), '\\|', ''), '|',
#               lda1000, '|', lda2000, '|', lda5000, '|', 
#               bizuinclass1, '|', bizuinclass2, '|', 
#               regexp_replace(titleseg, '\\|', ''), '|', 
#               regexp_replace(vtitleseg, '\\|', ''), '|', 
#               regexp_replace(contentseg, '\\|', ''), '|', 
#               firstorder, '|', secondorder)
  def processing_batch(self, lines, pred=False):
    addinfo = []
    label1 = []
    label2 = []
    lda1000 = []
    lda2000 = []
    lda5000 = []
    titleseg = []
    vtitleseg = []
    contentseg = []
    titlelen = []
    vtitlelen = []
    contentlen = []
    bizclass1 = []
    bizclass2 = []

    for line in lines:
      fields = line.strip().split('|')
      if len(fields) < 11 : continue

      addinfo.append(fields[0])
      lda1000.append(self.parseLDAArray(fields[1], 1000))
      lda2000.append(self.parseLDAArray(fields[2], 2000))
      lda5000.append(self.parseLDAArray(fields[3], 5000))
      if pred:
        bizclass1.append(self.parseBizClass(fields[4], self.level1_size, '10000'))
        bizclass2.append(self.parseBizClass(fields[5], self.level2_size, '10000'))        
      else:
        bizclass1.append(self.parseBizClass(fields[4], self.level1_size, fields[9]))
        bizclass2.append(self.parseBizClass(fields[5], self.level2_size, fields[10]))
      
      seg, slen = self.parseVocab(fields[6], self.titlemaxsize)
      titleseg.append(seg);titlelen.append(slen)
      seg, slen = self.parseVocab(fields[7], self.titlemaxsize)
      vtitleseg.append(seg);vtitlelen.append(slen)
      seg, slen = self.parseVocab(fields[8], self.articlemaxsize)
      contentseg.append(seg);contentlen.append(slen)
      
      label1.append(self.parseLable(fields[9], self.level1_size))
      label2.append(self.parseLable(fields[10], self.level2_size))
      

    return {'L': len(label1) if len(label1)==len(addinfo) else 0,
            'label1': label1, 'label2' : label2,
            'lda1000': lda1000, 'lda2000': lda2000, 'lda5000': lda5000,
            'bizclass1': bizclass1, 'bizclass2': bizclass2,
            'titleseg': titleseg, 'vtitleseg': vtitleseg, 'contentseg': contentseg,
            'titlelen': titlelen, 'vtitlelen': vtitlelen, 'contentlen': contentlen,
            'addinfo': addinfo }
                                         
  def read_preddata_batch(self, size=256):
    lines = self.preddata.read_batch(size)
    return self.processing_batch(lines)

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run Reindex.")
  parser.add_argument('--inputpath', default='data/', required=False,
                      help='Input data path.')
  parser.add_argument('--vocab', default='data/model2.vec.proc', required=False,
                      help='Vocab file path.')
  parser.add_argument('--emb_size', type=int, default=100,
                      help='Vocab file path.')
  parser.add_argument('--vocab_size', type=int, default=0,
                      help='Vocab size.')
  parser.add_argument('--dataset', nargs='+', default=['VedioClassifyBizuinInputAnsyEbm'],
                      help='Choose a train dataset.')
  parser.add_argument('--testset', nargs='*', default=['VedioClassifyBizuinInputAnsyEbm'],
                      help='Choose a test dataset.')
  parser.add_argument('--predset', nargs='*', default=['VedioClassifyBizuinInputAnsyEbm'],
                      help='Choose a pred dataset.')
  parser.add_argument('--shuffle_file', type=str2bool, default=True,
                      help='Suffle input file')
  parser.add_argument('--batch_size', type=int, default=5,
                      help='Data getbatch size')
  parser.add_argument('--batch_size_test', type=int, default=5,
                      help='Data getbatch size')
  args = parser.parse_args()
  print(vars(args))
  readdata = VedioClassifyBizuinInputAnsyEmb(vars(args))

  ret = readdata.read_preddata_batch(size=3)
  if ret['L']>0:
    print(ret)

#  readdata.start_ansyc()
#  train_data = readdata.read_testdata_batch_ansyc()
#  print(train_data)
##  print(train_data)
##  print(train_data['Q'][0].shape)
##  print(train_data['D'][0].shape)
##  train_data = readdata.read_traindata_batch_ansyc()
##  print(train_data)
#  readdata.stop_and_wait_ansyc()


