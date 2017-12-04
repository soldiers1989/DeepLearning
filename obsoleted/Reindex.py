#-*- coding: utf-8 -*-

import os
import argparse
import pickle

class ReindexData(object):
  def __init__(self, inputargs):
    self.args = inputargs
    self.inputpath = args.inputpath     #输入目录
    self.outputpath = args.outputpath   #输出目录
    self.inputfiles = args.inputfiles   #待处理文件列表
    self.postfix = args.postfix         #处理结果后缀
    self.remove_embedding = args.remove_embedding #删除embedding

    self.idmap = {}
    self.maxidx = 0
    self.appendidx = args.appendidx
    
    self.skipidxline = 0
    self.skipidxfield = 0
    self.skipembeddingfield = 0
    self.totalline = 0
    self.totalfield = 0

    self.idxfile = args.idxfile
    self.idxfilename = self.inputpath + '/' + self.idxfile
    if args.loadidxfile:
      if os.path.isfile(self.idxfilename):
        self.loadidmap()
        print("index file " + self.idxfilename + " Loaded")
      else:
        print("index file "+self.idxfilename+" NOT EXISTED")
    else:
      self.appendidx = True             #如果不加载index文件
    
    print("self.appendidx: "+str(self.appendidx))

  def load_data_from_file(self, file):
    with open(file, 'r') as rd:
      while True:
        line = rd.readline()
        if not line:
          break
        if '#' in line:
          punc_idx = line.index('#')
        else:
          punc_idx = len(line)
        label = float(line[0:1])
        if label>1:
          label=1
        feature_line = line[2:punc_idx]
        words = feature_line.split(' ')
        cur_feature_list = []
        for word in words:
          if not word:
            continue
          tokens = word.split(':')

          if len(tokens[1]) <= 0:
            tokens[1] = '0'
          cur_feature_list.append([int(tokens[0]), float(tokens[1])])
        yield label, cur_feature_list

  def reidx(self, flist):
    it = iter(flist)
    recline=True    
    self.totalline = self.totalline+1
    while True:
      try:
        idx, value = next(it) # 获得下一个值
        if value==0 : continue
        self.totalfield = self.totalfield + 1
        
        if idx>=406552 and self.remove_embedding:
        	self.skipembeddingfield = self.skipembeddingfield + 1
        	continue  # 不再添加index，不处理这个field
        
        if idx not in self.idmap:
          if not self.appendidx:
            self.skipidxfield = self.skipidxfield+1
            if recline:
              recline=False
              self.skipidxline=self.skipidxline+1
            continue  # 不再添加index，不处理这个field

          self.maxidx=self.maxidx+1
          self.idmap[idx]=self.maxidx
        yield (self.idmap[idx], value)
      except StopIteration:
        break # 遇到StopIteration就退出循环

  def run(self):
    if not os.path.exists(self.inputpath):
      print('Input dir "' + self.inputpath + '" NOT EXISTED')
      exit
    files = self.inputfiles.split('#')
    for f in files:
      fname = self.inputpath + '/' + f
      if not os.path.isfile(fname):
        print('Input file "'+fname+'" NOT EXISTED')
        continue
        
      outfname = self.outputpath + '/' + f +'.' +self.postfix
      with open(outfname, 'w') as outf:
        for label, feature_list in self.load_data_from_file(fname):
          reidx_feature_list=sorted(self.reidx(feature_list), key=lambda a: a[0], reverse=False)
          if len(reidx_feature_list)>0:
            reidx_feature_list=' '.join([str(item[0])+':'+str(item[1]) for item in reidx_feature_list])
            outf.write('1 ' if label > 0 else '0 ')
            outf.write(reidx_feature_list+'\n')

    self.saveidmap()
    print('Total line '+str(self.totalline)+', field: '+str(self.totalfield))
    print('Skip line '+str(self.skipidxline)+', field: '+str(self.skipidxfield)+', embedding: '+str(self.skipembeddingfield))

  def saveidmap(self):
    tosave = {"map":self.idmap, "maxidx":self.maxidx}
    pickle.dump(tosave, open(self.idxfilename, "w"))
    print('save '+self.idxfilename+', maxidx: '+str(self.maxidx))

  def loadidmap(self):
    toload = pickle.load(open(self.idxfilename, "r"))
    self.idmap = toload['map']
    self.maxidx = toload['maxidx']
    print('load '+self.idxfilename+', maxidx: '+str(self.maxidx))

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run Reindex.")
  parser.add_argument('--inputpath', default='data/', required=True,
            help='Input data path.')
  parser.add_argument('--outputpath', default='data/', required=True,
            help='Output data path.')
  parser.add_argument('--inputfiles', nargs='?', default='', required=True,
            help='Choose inputfiles, seperated by #.')
  parser.add_argument('--postfix', default='ridx',
            help='Reindex postfix.')
  parser.add_argument('--idxfile', default='ridxfile',
            help='index mapping file.')
  parser.add_argument('--loadidxfile', type=str2bool, default=False,
            help='Loading existed index.')
  parser.add_argument('--appendidx', type=str2bool, default=False,
            help='Add new index. If idxfile NOT EXISTED, it will set to True')
  parser.add_argument('--remove_embedding', type=str2bool, default=False,
            help='Remove Embedding Feature')
  args = parser.parse_args()
  reindexdata = ReindexData(args)
  reindexdata.run()