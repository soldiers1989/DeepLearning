#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pickle
import random
import datetime
import threading
import codecs
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

class VedioMatchInputAnsy(object):
  def __init__(self, inputargs):
    self.args = inputargs
    self.inputpath = inputargs['inputpath']
    
    self.userattr_size = inputargs['userattr_size']
    self.vedioattr_size = inputargs['vedioattr_size']

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

  def processing_doc(self, line, dim):
    ret = np.zeros(dim, dtype=float)
    fields = line.strip().split(' ')
    for item in fields:
      int_val =  int(item)
      if int_val<=0 or int_val>=dim: continue
      ret[int_val]=1.0

    return ret

  def parseFixArray(self, line, length):
    fields = line.strip().split(' ')
    ret = np.zeros(length)
    if len(fields)==length:
      for ii in range(length):
        ret[ii]=int(fields[ii])
    return ret

  def parseVarArray(self, line, maxid=9223372036854775807):
    ret = [int(item) for item in line.strip().split(' ')]
    filter( lambda x:x<maxid, ret )
    if( len(ret)==0 ): ret=[0]
    return ret
    
  def varArray2SparseTensorValue(self, arr, dim):
    indices=[]
    values=[]
    shape=[len(arr), dim]
    idx=0
    for arr1 in arr:
      for item in arr1:
        indices.append([idx,item])
        values.append(item)
        
      idx+=1
    return (indices, values, shape)

  def processing_batch(self, lines):
    user_base = []
    user_cctype = []
    user_cclassid1 = []
    user_cclassid2 = []
    user_ctag = []
    pos_vinfo = []
    pos_ctag = []
    neg_vinfo = []
    neg_ctag = []
    raw_data = []

    idx=0
    for line in lines:
      fields = line.strip().split(',')
      if len(fields) < 9: continue

      user_base.append(self.parseFixArray(fields[0],4))
      
      user_cctype.append(self.parseVarArray(fields[1]))
      user_cclassid1.append(self.parseVarArray(fields[2]))
      user_cclassid2.append(self.parseVarArray(fields[3]))
      user_ctag.append(self.parseVarArray(fields[4], self.vedioattr_size))
      
      pos_vinfo.append(self.parseFixArray(fields[5],3))
      
      pos_ctag.append(self.parseVarArray(fields[6], self.vedioattr_size))
      
      neg_vinfo.append(self.parseFixArray(fields[7],3))
      
      neg_ctag.append(self.parseVarArray(fields[8], self.vedioattr_size))
      raw_data.append(fields[9])
      
      idx+=1

      #1 10 15 431,
      #8 13 10 18 15 16 22 1 2 5 4 7,
      #146 61 38 178 187 82 136 211 205 48 174 216 32 71 221,
      #56 176 222 73 54 87 68 183 155 182 209 185 74 189 88 40 139 215 35 51 33 225 224 177 52 75 217 151 66,
      #4408 422 863 1380 1234 385 657 5553 601 7207 635 1461 4614 3330 4927 1978 5600 305 693 5130 304 1363 6031 2718 7080 982 3088 1131 1636 7478 348 842 6023 3629 762 7516 1341 1020 2968 1624 328 1425 525 1278 461 388 262 261 723 257 1343 771 301 608 1150 320 390 652 3192 2949 1265 1058 2200 889 6544 1752 1214 786 609 3625 510 5131 1632 391 1250 7596 2716 6967 290 2574 1240 543 2394 1431 4463 6020 6145 3551 238 921 550 2372 251 787 454 1153 3342 2988 4357 2415 4166 639 677 583 479 3606 4066 591 5260 488 392 1389 253 508 6732 430 4095 3759 1434 745 343 1620 573 2187 945 5294 244 377 717 903 6569 1954 681 271 254 3582 6244 2227 255 646 421 7870 259 621 237 3897 331 627 502 3643 242 440 2763 2152 1640 1745 1960 1267 1805 605 3837 329 729 3546 1114 3337 245,
      #0 123 134,
      #3081 356 4206 1025,
      #0 123 134,
      #946 792 1880 356 785 959,
      #1199578362/v0620xteldv/李嘉诚开记者会，突然李泽钜手机响了，注意听，铃声是最大亮点/u0612wsay1s/大学生向王健林借钱！王健林霸气反击：你有能耐见到我|1/1 41/7 4/2 2504/415|10/8 26/13 23/10 31/18 28/15 29/16 43/22 1/1 2/2 5/5 4/4 9/7 |1494462/115/146 1538931/30/61 1494464/7/38 1494460/147/178 1494467/156/187 1494446/51/82 1494448/105/136 1494454/180/211 1494455/174/205 1494456/17/48 1494457/143/174 1494465/185/216 1494463/1/32 1494452/40/71 1494466/190/221 |1494476/25/56 1494530/145/176 1494536/191/222 1497226/42/73 1494510/23/54 1497091/56/87 1497094/37/68 1494553/152/183 1494555/124/155 1494556/151/182 1494498/178/209 1528263/154/185 1494541/43/74 1538940/158/189 1528260/57/88 1494428/9/40 1494540/108/139 1494527/184/215 1494426/4/35 1494504/20/51 1494422/2/33 1494564/194/225 1494565/193/224 1528265/146/177 1494560/21/52 1494561/44/75 1494563/186/217 1494485/120/151 1538936/35/66 |违章停车/4175/4408 情感/189/422 动物厮杀/630/863 越野车/1147/1380 田径/1001/1234 手机/152/385 考古/424/657 牙签/5320/5553 黑帮/368/601 性能改装/6974/7207 家庭伦理/402/635 交通法规/1228/1461 超车/4381/4614 超载/3097/3330 蟒蛇/4694/4927 女性安全/1745/1978 悍马/5367/5600 用车百科/72/305 古墓/460/693 冷知识/4897/5130 表演秀/71/304 Jeep/1130/1363 城管/5798/6031 篮球/2485/2718 生活趣闻/6847/7080 武林风/749/982 开盖妙招/2855/3088 超级工程/898/1131 印度游/1403/1636 技巧/7245/7478 涨姿势/115/348 建筑景观/609/842 王斑/5790/6023 社会事件/3396/3629 工程机械/529/762 转弯/7283/7516 爆笑体育/1108/1341 我伙呆/787/1020 狗血/2735/2968 下一站幸福/1391/1624 KO/95/328 追尾/1192/1425 国内奇趣/292/525 动态展示/1045/1278 玩车达人/228/461 都市/155/388 动物世界/29/262 农村剧/28/261 斯诺克/490/723 趣闻趣事/24/257 高速路/1110/1343 挖掘机/538/771 趣味实验/68/301 鞭腿/375/608 汽车花边/917/1150 科技资讯/87/320 摄影/157/390 炫酷改装/419/652 倒车/2959/3192 花豹/2716/2949 大象/1032/1265 抢劫/825/1058 逆行/1967/2200 搞笑恶搞/656/889 水电站/6311/6544 魔术/1519/1752 碰瓷/981/1214 货车/553/786 搞笑段子/376/609 大桥/3392/3625 改装车/277/510 交通/4898/5131 美国游/1399/1632 约会相亲/158/391 无人机/1017/1250 双节棍/7363/7596 宋朝/2483/2716 岩浆/6734/6967 情感纠纷/57/290 恋恋不忘/2341/2574 社会调查/1007/1240 非诚勿扰/310/543 黄梦莹/2161/2394 丁俊晖/1198/1431 小伙子/4230/4463 反舰导弹/5787/6020 看鉴/5912/6145 车祸盘点/3318/3551 国内/5/238 犯罪/688/921 民间高手/317/550 翻车事故/2139/2372 搞怪妹子/18/251 东北人/554/787 航拍/221/454 鳄鱼/920/1153 碾压/3109/3342 撞车/2755/2988 武僧一龙/4124/4357 魔术教学/2182/2415 性生活/3933/4166 美景/406/639 老司机/444/677 航空母舰/350/583 自然风光/246/479 超速/3373/3606 交通事故集锦/3833/4066 女司机/358/591 液态铝/5027/5260 意外事件/255/488 雷人囧事/159/392 西瓜/1156/1389 韩国/20/253 厉害了我的哥/275/508 半挂车/6499/6732 前沿技术/197/430 奇闻趣事/3862/4095 性教育/3526/3759 猎食/1201/1434 低级趣味/512/745 奇葩/110/343 拜金女/1387/1620 科学趣闻/340/573 中日搏击/1954/2187 卡车/712/945 乞丐/5061/5294 夫妻剧/11/244 交通安全/144/377 科学奇闻/484/717 警匪/670/903 奇趣大自然/6336/6569 魔术揭秘/1721/1954 丰田/448/681 因吹斯汀/38/271 机械设备/21/254 趣闻实验/3349/3582 户外运动/6011/6244 制造技术/1994/2227 交通事故/22/255 武术/413/646 用车知识/188/421 不明生物/7637/7870 外国人/26/259 街访/388/621 车祸/4/237 于明加/3664/3897 重拳/98/331 台球/394/627 越野/269/502 桂林/3410/3643 生活小妙招/9/242 自由搏击/207/440 段子剧/2530/2763 经典传奇/1919/2152 老鼠/1407/1640 桥/1512/1745 恋情/1727/1960 魔术表演/1034/1267 近景魔术/1572/1805 中国制造/372/605 郭晋安/3604/3837 国际/96/329 港剧/496/729 自燃/3313/3546 中国功夫/881/1114 广西游/3104/3337 SUV车型/12/245 |||0/0 1494469/92 1494586/103|李嘉诚/2848/3081 企业家/123/356 长江实业/3973/4206 采访/792/1025 |||0/0 1494469/92 1494586/103|王健林/713/946 撒贝宁/559/792 万达/1647/1880 企业家/123/356 演讲/552/785 开讲啦/726/959

    return {'L': len(user_base),
            'user_base': user_base,
            'user_cctype':  self.varArray2SparseTensorValue( user_cctype, self.vedioattr_size),
            'user_cclassid1': self.varArray2SparseTensorValue( user_cclassid1, self.vedioattr_size),
            'user_cclassid2': self.varArray2SparseTensorValue( user_cclassid2, self.vedioattr_size),
            'user_ctag': self.varArray2SparseTensorValue( user_ctag, self.vedioattr_size),
            'pos_vinfo': pos_vinfo,
            'pos_ctag': self.varArray2SparseTensorValue( pos_ctag, self.vedioattr_size),
            'neg_vinfo': neg_vinfo,
            'neg_ctag': self.varArray2SparseTensorValue( neg_ctag, self.vedioattr_size),
            'raw_data': raw_data }

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
      if self.verbose==2: print(logingfo)
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
      if self.verbose==2: print(logingfo)
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
    queries = []
    docs = []
    labels = []
    pred = []
    lines = self.preddata.read_batch(size)
    for line in lines:
      fields = line.strip().split(',')
      if len(fields) < 3: continue

      queries.append(self.processing_doc(fields[0], self.dimquery))
      docs.append(self.processing_doc(fields[1], self.dimdoc))
      labels.append(int(fields[2]))

      if len(fields) == 4:
        pred.append(fields[3])

    return {'Q': queries, 'D': docs, 'Y': labels, 'L': len(queries), 'ID': pred }

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
  parser.add_argument('--dataset', nargs='+', default=['part-00000', 'part-00000'],
                      help='Choose a train dataset.')
  parser.add_argument('--testset', nargs='*', default=['part-00000', 'part-00000'],
                      help='Choose a test dataset.')
  parser.add_argument('--predset', nargs='*', default=['part-00000', 'part-00000'],
                      help='Choose a pred dataset.')
  parser.add_argument('--shuffle_file', type=str2bool, default=True,
                      help='Suffle input file')
  parser.add_argument('--batch_size', type=int, default=2,
                      help='Data getbatch size')
  parser.add_argument('--batch_size_test', type=int, default=3,
                      help='Data getbatch size')
  parser.add_argument('--userattr_size', type=int, default=706,  #13715
                      help='User attr dim')
  #parser.add_argument('--vedioattr_size', type=int, default=8368,  #12774
  parser.add_argument('--vedioattr_size', type=int, default=4855, 
                      help='Vedio attr dim')
  parser.add_argument('--pred', default='query', required=False,
                      help='Witch part to pred')
  args = parser.parse_args()
  print(vars(args))
  readdata = VedioMatchInputAnsy(vars(args))

#  ret = readdata.read_preddata_batch(size=3)
#  if ret['L']>0:
#    print(ret)
#    #print(ret['Q'])
#
#  while ret['L']>0:
#    #print(ret['L'])
#    print(ret)
#    ret = readdata.read_preddata_batch(size=3)
#  print('*'*20)
#  ret = readdata.read_preddata_batch(size=3)
#  print(ret)

  readdata.start_ansyc()
  train_data = readdata.read_testdata_batch_ansyc()
  print(train_data)
#  print(train_data)
#  print(train_data['Q'][0].shape)
#  print(train_data['D'][0].shape)
#  train_data = readdata.read_traindata_batch_ansyc()
#  print(train_data)
  readdata.stop_and_wait_ansyc()


