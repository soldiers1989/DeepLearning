# -*- coding: utf-8 -*-

import sys
import os
import pickle
import random
import datetime
import threading
import numpy as np
import argparse
from itertools import product
from itertools import islice
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
  def __init__(self, files, shuffleFile=True, shuffleRecord=True, epochs=None):
    self.capacity = 1
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
    #print('Processing ' + filename)
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

class MPInputAnsy(object):
  def __init__(self, inputargs):
    self.args = inputargs
    self.dim =  inputargs['dim']
    self.inputpath = inputargs['inputpath']

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

    self.traindata = TxtFilesConcurrentRandomReaderV2(inputargs)
    self.testdata = TxtFilesRandomReader(files=self.testset, shuffleFile=inputargs['shuffle_file'])
    self.preddata = TxtFilesRandomReader(files=self.predset, shuffleFile=False, shuffleRecord=False,
                                         epochs=1)

    self.ansy_run = True
    self.traindataqueue = Queue(maxsize=5)
    self.testdataqueue = Queue(maxsize=3)
    self.readlock = threading.Lock()
    self.threads = []
    
  def processing_doc(self, line):
    ret = np.zeros(self.dim, dtype=float)
    fields = line.strip().split(' ')
    for item in fields:
      int_val =  int(item)
      if int_val<=0 or int_val>=self.dim: continue
      ret[int_val]=1.0
      
    return ret

  def processing_batch(self, lines):
    queries = []
    docs = []
    labels = []
    for line in lines:      
      fields = line.strip().split(',')
      if len(fields) != 3: continue
      
      queries.append(self.processing_doc(fields[0]))
      docs.append(self.processing_doc(fields[1]))
      labels.append(int(fields[2]))

    return {'Q': queries, 'D': docs, 'Y': labels, 'L': len(queries)}

  def read_traindata_batch(self, size=256):
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
      print(logingfo)
    print('End thread for traindata.read_batch')

  def do_ansyc_testset(self):
    print('Begin thread for traindata.read_batch')
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
    self.preddata = TxtFilesRandomReader(files=self.predset, shuffleFile=False, shuffleRecord=False,
                                         epochs=1)

  def parse_data(self, line):
    fields = line.strip().split()
    label = int(fields[0])

    feaures = [(item.split(':')) for item in fields[1:]]
    feaures = sorted(filter(lambda x: x[1] != 0, [(int(item[0]), float(item[1])) for item in feaures]),
                     key=lambda a: a[0])
    return label, feaures

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
  parser.add_argument('--dataset', nargs='+', default=['sample.bincai.txt.num', 'sample.bincai2.txt.num'],
                      help='Choose a train dataset.')
  parser.add_argument('--testset', nargs='*', default=[],
                      help='Choose a test dataset.')
  parser.add_argument('--predset', nargs='?', default='',
                      help='Choose a pred dataset.')
  parser.add_argument('--shuffle_file', type=str2bool, default=True,
                      help='Suffle input file')
  parser.add_argument('--batch_size', type=int, default=2,
                      help='Data batch size')
  #parser.add_argument('--dim', type=int, default=12540,
  parser.add_argument('--dim', type=int, default=64,
                      help='Data dim')
  args = parser.parse_args()
  print(vars(args))
  readdata = MPInputAnsy(vars(args))
  
  readdata.start_ansyc()
  train_data = readdata.read_traindata_batch_ansyc()  
  print(train_data)
  train_data = readdata.read_traindata_batch_ansyc()  
  print(train_data)
  readdata.stop_and_wait_ansyc()


