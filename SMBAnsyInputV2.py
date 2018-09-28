#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
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
    for f in args.get('dataset', []):
      fname = f
      if not os.path.isfile(fname):
        fname = self.inputpath + '/' + f
      if not os.path.isfile(fname):
        print('*'*10+' Input file "' + fname + '" NOT EXISTED')
        continue
      fsize = os.path.getsize(fname)
      self.totalfilesize += fsize
      if fsize < filesizemin: filesizemin = fsize
      tmpfiles.append([fname, fsize])
    [ii.append(round(float(ii[1]) * 1000 / filesizemin)) for ii in tmpfiles]
    self.files = tmpfiles
    print('TxtFilesConcurrentRandomReaderV2:' + str(self.files))

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
      # fd = open(f[0], 'r')
      fd = codecs.open(f[0], 'r', 'utf-8')
      self.openfile.append((fd, f[2] * 1.0 / sumprob))
      self.readfileprobs.append(f[2] * 1.0 / sumprob)
    if self.verbose == 2: print(self.openfile)
    if self.verbose == 2: print(self.readfileprobs)

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

  def _next_line(self):  # 从某一个文件中读取一行
    while True:
      idx, f = next(self.nextfile)
      line = f[0].readline()
      if not line:
        self.closeopenfile(idx)
      else:
        yield line.strip()

  def _next_file_batch(self, size=0):  # 从某一个文件中读取size行
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
    self.epochs = epochs  # epochs=None会不停地读，不结束
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
    print('Processing ' + filename)
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

class SMBInputData(object):
  def __init__(self, line):
    # 815470608|3|
    # w07004o8k5u|多年未清理的水坑抽干水，发现罕见的大鱼，引全村人来围观|1530844127|98|4583|756 2860 43687 219692 149 4038 5934 32737 3329|46|12646 1860 47061|
    # l0527crq5q4|水库放了几天的水，终于把鱼王给盼了出来，在场的人可乐坏了|1530844231|98|2849|7628 1533 453 129800 229 6826 41264|35|1329 12646 7628|
    # l0561sbl5an|水库放水终于可以捕大鱼啦，爆爽|1530844234|98|2849|7628 22596 453 22 5934 277914|29|12646 7628
    self.line = line
    self.splited = line.split('|')  # 9n+2
    self.ok = True if len(self.splited) > 10 and (len(self.splited) - 2) % 9 == 0 else False
    self.uin = int(self.splited[0]) if self.ok else 0
    self.totalfields = int(self.splited[1]) if self.ok else 0
    self.readfield = 0
    self.fields = []

    try:
      for ii in range(self.totalfields):
        vid, vnid, vt, ts, c1, c2, vtn, _, tag = self.splited[2 + ii * 9:2 + (1 + ii) * 9]
        if len(c1.strip()) == 0: c1 = '0'
        if len(c2.strip()) == 0: c2 = '0'
        if len(vtn.strip()) == 0: vtn = '0'
        if len(tag.strip()) == 0: tag = '0'
        vtn = [int(x) for x in vtn.strip().split(' ')]
        tag = [int(x) for x in tag.strip().split(' ')]
        self.fields.append((vid, int(vnid), vt, int(ts), int(c1), int(c2), vtn, tag))
    except Exception as e:
      print('str(Exception):\t'+str(Exception))
      print('str(e):\t'+str(e))
      print('repr(e):\t'+str(repr(e)))
      print('e.line:\t'+line)
      self.ok=False
      self.totalfields=0

  def nexttrainitem(self):
    if self.ok and self.readfield < self.totalfields - 1:
      self.readfield += 1
      return (self.fields[self.readfield - 1], self.fields[self.readfield])
    return None
    
  def nextpreditem(self):
    if self.ok and self.readfield < self.totalfields:
      self.readfield += 1
      return self.fields[self.readfield - 1]
    return None

class SMBAnsyInputV2(object):
  def __init__(self, inputargs):
    self.args = inputargs
    print('SMBAnsyInput:', str(self.args))
    self.inputpath = inputargs['inputpath']
    if not self.inputpath.endswith(os.sep):
      self.inputpath += os.sep
    
    self.dataset = []
    self.testset = []
    self.predvedioset = []
    self.preduserset = []

    if 'dataset' in inputargs:
      self.dataset = [self.inputpath + item for item in inputargs['dataset']]
      print('self.dataset %s' % self.dataset)
    if 'testset' in inputargs:
      self.testset = [self.inputpath + item for item in inputargs['testset']]
      print('self.testset %s' % self.testset)
    if 'predvedioset' in inputargs:
      self.predvedioset = [self.inputpath + item for item in inputargs['predvedioset']]
      print('self.predvedioset %s' % self.predvedioset)
    if 'preduserset' in inputargs:
      self.preduserset = [self.inputpath + item for item in inputargs['preduserset']]
      print('self.preduserset %s' % self.preduserset)

    self.vocab = Vocab(inputargs['vocab'], inputargs['emb_size'], inputargs['vocab_size'])
    self.vocab_size = inputargs['vocab_size']
    print('*'*20 + 'self.vocab_size %d' % self.vocab_size )
    self.titlemaxsize = 20
    self.articlemaxsize = 200

    self.traindata = TxtFilesConcurrentRandomReaderV2(inputargs)

    self.testdata = TxtFilesRandomReader(files=self.testset, shuffleFile=inputargs.get('shuffle_file', True))

    self.predvediodata = TxtFilesRandomReader(files=self.predvedioset, shuffleFile=False, shuffleRecord=False,
                                         epochs=1)

    self.batch_size = inputargs.get('batch_size', 0)
    self.batch_size_test = inputargs.get('batch_size_test', self.batch_size * 2)
    
    self.ansy_run = True
    self.traindataqueue = Queue(maxsize=5)
    self.testdataqueue = Queue(maxsize=3)
    self.threads = []
    self.verbose = 0

    self.readlock = threading.Lock()

    self.items = []  
    self.defaultuseritem = None 
    self.useritems = []
    
    if len(self.preduserset)>0:
      self.reset_preduserset()

  def do_ansyc_trainset(self):
    print('Begin thread for traindata.read_batch')

    for ii in range(self.batch_size):
      with self.readlock:
        sdata = SMBInputData(self.traindata.read_batch(1)[0])
        while not sdata.ok:
          print('IN while not sdata.ok')
          sdata = SMBInputData(self.traindata.read_batch(1)[0])
        self.items.append(sdata)

    while self.ansy_run:
      self.traindataqueue.put(self.processing_batch_train())

    print('End thread for traindata.read_batch')

  def do_ansyc_testset(self):
    print('Begin thread for testdata.read_batch')
    if self.has_testset():
      while self.ansy_run:
        with self.readlock:
          lines = self.testdata.read_batch(self.batch_size_test)
        self.testdataqueue.put(self.processing_batch_train())

    print('End thread for testdata.read_batch')

  def start_ansyc(self):
    self.threads = []
    if len(self.dataset) > 0:
      t = threading.Thread(target=self.do_ansyc_trainset)
      self.threads.append(t)
      t.start()
    else:
      print('*' * 20 + 'self.dataset is NULL' + '*' * 20)

    if len(self.dataset) > 0:
      t = threading.Thread(target=self.do_ansyc_testset)
      self.threads.append(t)
      t.start()
    else:
      print('*' * 20 + 'self.dataset is NULL' + '*' * 20)

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

  def processing_item(self, item):    
    # (vid, vnid, vt, int(ts), int(c1), int(c2), vtn, tag)
    # ('w07004o8k5u', 8, '多年未清理的水坑抽干水，发现罕见的大鱼，引全村人来围观', 1530844127, 98, 4583, [756, 2860, 43687, 219692, 149, 4038, 5934, 32737, 3329], [12646, 1860, 47061])
    linelen = len(item[6])
    if (linelen < self.titlemaxsize): item[6].extend([0] * (self.titlemaxsize - linelen))
    cat1 = 0 if item[4] > self.vocab_size else item[4]
    cat2 = 0 if item[5] > self.vocab_size else item[5]
    title = [x if x <= self.vocab_size else 0 for x in item[6][:self.titlemaxsize]]
    return item[0] + '|' + item[2], item[1], [cat1, cat2], np.array(title), min(linelen, self.titlemaxsize)

  def processing_batch_train(self):
    Xlabel, Ylabel = [], []
    Xid, Yid = [], []
    Xbizclass, Ybizclass = [], []
    Xvtitleseg, Yvtitleseg = [], []
    Xvtitlelen, Yvtitlelen = [], []
    restart = []

    for ii in range(len(self.items)):
      restartf = 0
      item = self.items[ii].nexttrainitem()
      while item is None:
        with self.readlock:
          sdata = SMBInputData(self.traindata.read_batch(1)[0])
          while not sdata.ok:
            print('IN while not sdata.ok in processing_batch_train')
            sdata = SMBInputData(self.traindata.read_batch(1)[0])
          self.items[ii] = sdata
          item = self.items[ii].nexttrainitem()
          restartf = 1

      label, vnid, bizclass, vtitleseg, vtitlelen = self.processing_item(item[0])
      Xlabel.append(label); Xid.append(vnid); Xbizclass.append(bizclass);
      Xvtitleseg.append(vtitleseg); Xvtitlelen.append(vtitlelen)

      label, vnid, bizclass, vtitleseg, vtitlelen = self.processing_item(item[1])
      Ylabel.append(label); Yid.append(vnid); Ybizclass.append(bizclass);
      Yvtitleseg.append(vtitleseg); Yvtitlelen.append(vtitlelen)

      restart.append(restartf)

    return {'L': len(Xlabel) if len(Ylabel) == len(Xlabel) and len(Xlabel) == len(Yvtitlelen) else 0,
            'Xlabel': Xlabel, 'Ylabel': Ylabel,
            'Xid': Xid, 'Yid': Yid,
            'Xbizclass': Xbizclass, 'Ybizclass': Ybizclass,
            'Xvtitleseg': Xvtitleseg, 'Yvtitleseg': Yvtitleseg,
            'Xvtitlelen': Xvtitlelen, 'Yvtitlelen': Yvtitlelen,
            'restart': restart}
    
  def has_preduserset(self):
    return len(self.preduserset) > 0
    
  def get_predusernext(self):
    while True:
      data=self.preduserdata.read_batch(1)
      if len(data)==0: 
        if self.defaultuseritem is None: 
          return 2, None
        else:
          return 1, SMBInputData(self.defaultuseritem)
      else:
        sdata = SMBInputData(data[0])
        if sdata.ok:
          if self.defaultuseritem is None: 
            self.defaultuseritem = data[0]
          return 0, sdata

  def reset_preduserset(self):
    self.preduserdata = TxtFilesRandomReader(files=self.preduserset, shuffleFile=False, shuffleRecord=False,
                                         epochs=1)    
    for ii in range(self.batch_size):
      with self.readlock:
        scene, item = self.get_predusernext()
        if scene==2: break
        self.useritems.append((scene, item))
                                         
  def read_preduserdata_batch(self):    
    Xlabel, Xuin, Xid, Xbizclass, Xvtitleseg, Xvtitlelen, restart, Xpadding = [],[],[],[],[],[],[],[]
    allpaddingflag = 1

    for ii in range(len(self.useritems)):
      restartf = 0
      padding, item = self.useritems[ii]
      uin=item.uin
      item = item.nextpreditem()
      while item is None:
        with self.readlock:
          padding, item = self.get_predusernext()
          uin=item.uin
          self.useritems[ii] = (padding, item)
          item = item.nextpreditem()
          restartf = 1

      label, vnid, bizclass, vtitleseg, vtitlelen = self.processing_item(item)
      Xlabel.append(str(vnid)+'|'+label); Xuin.append(uin); Xid.append(int(vnid));
      Xbizclass.append(bizclass); Xvtitleseg.append(vtitleseg); Xvtitlelen.append(vtitlelen)
      restart.append(restartf), Xpadding.append(padding)
      
      if padding==0: allpaddingflag=0

    return {'L': len(Xlabel) if allpaddingflag==0 else 0,
            'label': Xlabel, 'uin': Xuin, 'Xid': Xid,
            'bizclass': Xbizclass, 'vtitleseg': Xvtitleseg, 'vtitlelen': Xvtitlelen,
            'restart': restart, 'padding': Xpadding}
              
  def has_predvedioset(self):
    return len(self.predvedioset) > 0

  def reset_predvedioset(self):
    self.predvediodata = TxtFilesRandomReader(files=self.predvedioset, shuffleFile=False, shuffleRecord=False,
                                         epochs=1)  
                                         
  def read_predvediodata_batch(self, size=256):
    Xlabel = []
    Xbizclass = []
    Xvtitleseg = []
    Xvtitlelen = []
    Xid = []
    
    lines = self.predvediodata.read_batch(size)
    for line in lines:      
      #a00162kms3x|20160|苏州好风光|0|220|1916|3995 4286|2|0
      #x066288y1or|1760|真的谁都希望自己越活越年轻|0|676|32360|163 191 27 70653 786|5|51070 40285
      fields = line.split('|')
      if len(fields)!=9: continue
        
      cat1 = int(fields[4])
      if cat1 > self.vocab_size: cat1=0
      cat2 = int(fields[5])
      if cat2 > self.vocab_size: cat2=0
        
      titleids=[int(x) for x in fields[6].strip().split(' ') if x.strip() != '']
      linelen=len(titleids)      
      if (linelen < self.titlemaxsize): titleids.extend([0] * (self.titlemaxsize - linelen))
      vtitleseg = [x if x <= self.vocab_size else 0 for x in titleids[:self.titlemaxsize]]
        
      Xlabel.append('|'.join(fields[0:3]))
      Xid.append(int(fields[1]))
      Xbizclass.append([cat1, cat2])
      Xvtitleseg.append(vtitleseg);
      Xvtitlelen.append(min(linelen, self.titlemaxsize))    
    
    return {'L': len(Xlabel) if len(Xlabel) == len(Xbizclass) and len(Xvtitleseg) == len(Xbizclass) else 0,
            'Xlabel': Xlabel, 'Xbizclass': Xbizclass, 'Xid': Xid,
            'Xvtitleseg': Xvtitleseg, 'Xvtitlelen': Xvtitlelen }

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
  parser.add_argument('--vocab_size', type=int, default=1000,
                      help='Vocab size.')
  parser.add_argument('--dataset', nargs='+', default=['smbtrainsample.v2', 'smbtrainsample.v2'],
                      help='Choose a train dataset.')
  parser.add_argument('--testset', nargs='*', default=['smbtrainsample.v2', 'smbtrainsample.v2'],
                      help='Choose a test dataset.')
  parser.add_argument('--predvedioset', nargs='+', default=['smbvedio.v2'],
                      help='Choose a pred dataset.')
  parser.add_argument('--preduserset', nargs='+', default=['smbuser.v2'],
                      help='Choose a pred dataset.')
  parser.add_argument('--shuffle_file', type=str2bool, default=True,
                      help='Suffle input file')
  parser.add_argument('--batch_size', type=int, default=3,
                      help='Data getbatch size')
  parser.add_argument('--batch_size_test', type=int, default=5,
                      help='Data getbatch size')
  args = parser.parse_args()
  print(vars(args))

  readdata = SMBAnsyInputV2(vars(args))
#  readdata.start_ansyc()
#  for ii in range(20):
#    train_data = readdata.read_traindata_batch_ansyc()
#    print(' ')
#    for jj in range(train_data['L']):
#      print('restart: %d %d %d'%(ii, jj, train_data['restart'][jj]))
#      print('Xlabel: %d %d %s'%(ii, jj, train_data['Xlabel'][jj]))
#      print('Ylabel: %d %d %s'%(ii, jj, train_data['Ylabel'][jj]))
#  readdata.stop_and_wait_ansyc()

#
#  ret = readdata.read_predvediodata_batch(size=3)
#  if ret['L']>0:
#    print(ret)

  ret = readdata.read_preduserdata_batch()
  while ret['L']>0:
    print(ret)
    ret = readdata.read_preduserdata_batch()

#  aSMBInputData = SMBInputData('815470608|3|w07004o8k5u|多年未清理的水坑抽干水，发现罕见的大鱼，引全村人来围观|1530844127|98|4583|756 2860 43687 219692 149 4038 5934 32737 3329|46|12646 1860 47061|l0527crq5q4|水库放了几天的水，终于把鱼王给盼了出来，在场的人可乐坏了|1530844231|98|2849|7628 1533 453 129800 229 6826 41264|35|1329 12646 7628|l0561sbl5an|水库放水终于可以捕大鱼啦，爆爽|1530844234|98|2849|7628 22596 453 22 5934 277914|29|12646 7628')
#  print('aSMBInputData '+str(aSMBInputData.ok))
#  print('aSMBInputData '+str(aSMBInputData.nextitem()))
#  print('aSMBInputData '+str(aSMBInputData.nextitem()))
#  print('aSMBInputData '+str(aSMBInputData.nextitem()))

