# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import pickle
import random
import re
import string
import sys
import threading
from itertools import islice


class TxtFileReader(object):
  def __init__(self, file):
    self.DEFAULT_BATCH_SIZE = 1024
    self.file = file
    self.nextline = self._read_line()

  def _read_line(self):  # 读一行的迭代器
    with open(self.file, 'r') as rd:
      while True:
        line = rd.readline()
        if not line: break
        yield line.strip()

  def read_line(self):  # 读一行
    return next(self.nextline)

  def read_batch(self, size=0):  # 读一个batch
    if size == 0: size = self.DEFAULT_BATCH_SIZE
    return list(islice(self.nextline, size))

  def stat_batch(self, size=0):
    if size == 0: size = self.DEFAULT_BATCH_SIZE
    idstat = {}
    cnt = 0
    totalfield = 0
    for line in self.read_batch(size):
      cnt += 1
      fields = line.split('#')
      if len(fields) != 3: continue

      for field in fields:
        wds = field.split(' ')
        for wd in wds:
          totalfield += 1
          if wd not in idstat:
            idstat[wd] = 1
          else:
            idstat[wd] += 1

    return cnt, totalfield, idstat


class StatData(object):
  def __init__(self, inputargs):
    self.args = inputargs
    self.inputpath = inputargs.inputpath  # 输入目录
    self.outputpath = inputargs.outputpath  # 输出目录
    self.dataset = inputargs.dataset  # 待处理文件列表

    self.idmaplock = threading.Lock()

    self.idstat = {}  # 词计数
    self.cidmap = {}  # 字符计数

    self.idmap = {}  # 下标到字符的映射
    self.cmap = {}  # 字符到下标的映射

    self.maxidx = 0

    self.totalline = 0
    self.totalfield = 0
    self.statfile = self.outputpath + '/' + inputargs.statfile

    self.negshuffle = []
    # self.negshufflemax = 50000
    self.negshufflemax = 5

  def filestat(self, fname):
    print("Processing (in thread):" + fname)  # self.statfile(fname)
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
      exit()

    threads = []
    for f in self.dataset:
      fname = self.inputpath + '/' + f
      if not os.path.isfile(fname):
        print('Input file "' + fname + '" NOT EXISTED')
        continue

      t = threading.Thread(target=self.filestat, args=(fname,))
      threads.append(t)
      t.start()

    for t in threads:
      t.join()

    self.createmoremap()
    self.saveidmap()
    print('Total line ' + str(self.totalline) + ', field: ' + str(self.totalfield))

  def createmoremap(self):
    self.cidmap = {}
    punc = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    punc = punc.decode("utf-8") + string.punctuation
    pitem = 0
    exclude = set(punc)
    for item in self.idstat:
      for k in item.decode('utf-8'):
        if k in exclude or ord(k) < 128:
          continue  # 去掉中文符号，英文数字单词符号

        v = self.idstat[item]
        if k not in self.cidmap:
          self.cidmap[k] = v
        else:
          self.cidmap[k] += v

      pitem += 1
      if pitem <= 10:
        print("item[%s]=[%d]" % (item, self.idstat[item]))

  def saveidmap(self):
    tosave = { "idstat": self.idstat,  # map
               "cidmap": self.cidmap,  # cidmap
               "idmap": self.idmap,    # idmap
               "cmap": self.cmap,      # cmap
               "maxidx": self.maxidx } # maxidx

    print('save ' + self.statfile + ', maxidx: ' + str(self.maxidx))
    pickle.dump(tosave, open(self.statfile, 'wb'))

    pitem = 0
    for item in self.cidmap:
      if pitem <= 10:
        print("item[%s]=[%d]" % (item, self.cidmap[item]))
        pitem += 1
      else:
        break

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
    self.idstat = toload['map'] if 'map' in toload else toload['idstat']
    self.cidmap = toload['cidmap'] if 'cidmap' in toload else {}
    self.idmap = toload['idmap'] if 'idmap' in toload else {}
    self.cmap = toload['cmap'] if 'cmap' in toload else {}
    self.maxidx = toload['maxidx'] if 'maxidx' in toload else 0

    if len(self.idmap) == 0:
      self.cmap = {}
      for idx, data in enumerate(sorted(self.cidmap.items(), lambda x, y: x[1]<y[1], reverse=True), start=1):
        self.idmap[idx] = data[0]
        self.cmap[data[0]] = idx

    tosave = {"idmap": self.idmap,  # idmap
              "cmap": self.cmap}  # maxidx

    pickle.dump(tosave, open(self.statfile + '.vocab', 'wb'))

    tosave = {"idstat": self.idstat,  # map
              "cidmap": self.cidmap,  # cidmap
              "idmap": self.idmap,  # idmap
              "cmap": self.cmap,  # cmap
              "maxidx": self.maxidx}  # maxidx
    print('save ' + self.statfile + ', maxidx: ' + str(self.maxidx))
    pickle.dump(tosave, open(self.statfile, 'wb'))

  def readdatafilebatch(self, tfr):
    rline = 0
    ret = []
    for line in tfr.read_batch():
    	#0
    	#20180413
    	#2636903641
    	#13030936711307172552
    	#r0560yseg1d
    	#1
    	#26
    	#3
    	#3209
    	#1
    	#4
    	#1494446
    	#1494483
    	#象棋教学;象棋;棋牌;残局
    	#2
    	#0:3;1:3;22:1;29:1;31:2;3:3;43:5;4:1;6:7
    	#1494446:1;1494447:2;1494449:5;1494452:5;1494460:1;1494461:7;1494462:1;1494463:3;1494468:1#1494423:3;1494437:2;1494439:3;1494483:1;1494519:5;1494520:1;1494546:1;1494553:1;1494570:1;1494576:1;1494578:1;1494589:5;1543614:1
    	#2333:1;MMORPG:5;mad:1;不忠:1;全智贤:1;冯提莫:1;海贼王:5;几分钟看电影:3;出轨:1;王者荣耀:1;刀剑血拼:1;奇幻:2;刀工:1;剧情:1;功夫格斗:1;跑车:1;动作冒险:2;大话西游2:3;新大话西游2:2;爆笑婚礼:1;夫妻:1;奇趣:2;夺宝联盟:1;女主播:1;热舞:1;妮可·罗宾:1;威航:1;少年:3;布加迪:1;性感:1;性感美女:1;恶搞网管:1;我伙呆:3;搞笑:2;每天一部好电影:1;托尼·贾:1;抢劫偷盗:1;美国:1;抽奖:1;拳霸2:1;拳霸:1;撩妹:1;整蛊:1;日漫:3;棋牌:1;豪车展示:1;武则天:1;残局:1;民间高手:1;泰国:1;海草舞:2;醉酒:1;涨姿势:1;游戏杂谈:1;游戏试玩:1;熟人整蛊:1;爱情:1;爱情鸡汤:1;牛人:1;狗血爱情:1;糗事:2;超能力者:1;索隆:1;经典动画:2;绝活:1;网红:1;韩国:1;节目解说:5;萨博:1;虚荣:1;象棋:1;象棋教学:1;路飞:1;跳舞:1;身材火辣:1;阿七解说:1

      fields = line.split('#')
      if len(fields) != 3: continue

      ret.append(fields)
      rline += 1

    return rline, ret

  def putneg(self, data):
    with self.idmaplock:
      for item in data:
        if len(self.negshuffle) < self.negshufflemax:
          self.negshuffle.append((item[0], item[1]))
        else:
          self.negshuffle[random.randint(0, self.negshufflemax - 1)] = (item[0], item[1])

      for item in data:
        nitem = self.negshuffle[random.randint(0, len(self.negshuffle) - 1)]
        while nitem[0] == item[0]:
          nitem = self.negshuffle[random.randint(0, len(self.negshuffle) - 1)]
        item.append(nitem[1])

    return data

  def createdatafile(self, fname):
    outfname = self.outputpath + '/' + fname + '.' + self.args.postfix
    outfnamenum = self.outputpath + '/' + fname + '.' + self.args.postfixnum
    fname = self.inputpath + '/' + fname
    tfr = TxtFileReader(fname)
    with open(outfname, 'w') as outf, open(outfnamenum, 'w') as outnumf:
      rline, data = self.readdatafilebatch(tfr)
      while rline > 0:
        data = self.putneg(data)

        for item in data:
          item0 = set([self.cmap.get(k, 0) for k in item[0].decode('utf-8')])
          item0.add(0)
          item0.remove(0)
          if (len(item0) < 10): continue
          item0 = ' '.join([str(k) for k in item0])

          item1 = set([self.cmap.get(k, 0) for k in item[1].decode('utf-8')])
          item1.add(0)
          item1.remove(0)
          if (len(item1) < 10): continue
          item1 = ' '.join([str(k) for k in item1])

          item3 = set([self.cmap.get(k, 0) for k in item[3].decode('utf-8')])
          item3.add(0)
          item3.remove(0)
          if (len(item3) < 10): continue
          item3 = ' '.join([str(k) for k in item3])

          outnumf.write(','.join((item0, item1, '1')))
          outnumf.write('\n')
          outnumf.write(','.join((item0, item3, '0')))
          outnumf.write('\n')

          #######################

          outf.write(','.join((item[0], item[1], '1')))
          outf.write('\n')
          outf.write(','.join((item[0], item[3], '0')))  # 没有用2
          outf.write('\n')

        rline, data = self.readdatafilebatch(tfr)

  def createdata(self):
    if not os.path.exists(self.inputpath):
      print('Input dir "' + self.inputpath + '" NOT EXISTED')
      exit()

    if not os.path.isfile(self.statfile):
      print('Input file "' + self.statfile + '" NOT EXISTED')
      exit()

    self.loadidmap()

    threads = []
    for f in self.dataset:
      fname = self.inputpath + '/' + f
      if not os.path.isfile(fname):
        print('Input file "' + fname + '" NOT EXISTED')
        continue

      print("processing:" + fname)  # self.statfile(fname)
      t = threading.Thread(target=self.createdatafile, args=(f,))
      threads.append(t)
      t.start()

    for t in threads:
      t.join()


def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
  Py3 = sys.version_info[0] == 3
  if not Py3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

  cmdinfo = '\n  statdata: stat files'
  cmdinfo += '\n  createdata: create train data'
  cmdinfo += '\n  praw: create predict data'
  parser = argparse.ArgumentParser(usage='%(prog)s [options]' + cmdinfo)
  subparsers = parser.add_subparsers()

  # 统计数据
  stat_data = subparsers.add_parser('statdata')
  stat_data.add_argument('--inputpath', default='data/', required=False,
                         help='Input data path.')
  stat_data.add_argument('--dataset', nargs='+', default=['sample.bincai.txt', 'sample.bincai2.txt'],
                         help='Choose a train dataset.')
  stat_data.add_argument('--outputpath', default='data/', required=False,
                         help='Output data path.')
  stat_data.add_argument('--statfile', default='statfile', required=False,
                         help='stat mapping file.')
  stat_data.set_defaults(func=lambda args: StatData(args).statdata())

  # 生成训练数据
  reindex_data = subparsers.add_parser('createdata')
  reindex_data.add_argument('--inputpath', default='data/', required=False,
                            help='Input data path.')
  reindex_data.add_argument('--dataset', nargs='+', default=['sample.bincai.txt', 'sample.bincai2.txt'],
                            help='Choose data to process.')
  reindex_data.add_argument('--outputpath', default='data/', required=False,
                            help='Output data path.')
  reindex_data.add_argument('--postfix', default='processed',
                            help='Reindex postfix.')
  reindex_data.add_argument('--postfixnum', default='num',
                            help='Reindex postfix.')
  reindex_data.add_argument('--statfile', default='statfile', required=False,
                            help='stat mapping file.')
  reindex_data.add_argument('--vocalsize', type=int, default=100000, required=False,
                            help='stat mapping file.')
  reindex_data.set_defaults(func=lambda args: StatData(args).createdata())

  #raw data生成预测数据
  reindex_rawdata = subparsers.add_parser('praw')
  reindex_rawdata.add_argument('--inputpath', default='data/', required=False,
                            help='Input data path.')
  reindex_rawdata.add_argument('--dataset', nargs='+', default=['sample.bincai.txt', 'sample.bincai2.txt'],
                            help='Choose data to process.')
  reindex_rawdata.add_argument('--outputpath', default='data/', required=False,
                            help='Output data path.')
  reindex_rawdata.add_argument('--postfix', default='processed',
                            help='Reindex postfix.')
  reindex_rawdata.add_argument('--postfixnum', default='num',
                            help='Reindex postfix.')
  reindex_rawdata.add_argument('--statfile', default='statfile', required=False,
                            help='stat mapping file.')
  reindex_rawdata.add_argument('--vocalsize', type=int, default=100000, required=False,
                            help='stat mapping file.')
  reindex_rawdata.add_argument('--idffile', default='idffile', required=False,
                            help='idf  file.')
  reindex_rawdata.set_defaults(func=lambda args: StatData(args).praw())

  args = parser.parse_args()

  print('args: ' + str(args))
  args.func(args)
