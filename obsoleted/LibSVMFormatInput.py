#-*- coding: utf-8 -*-

import numpy as np
import random
from itertools import islice

class LoadLibSvmData(object):
  # Three files are needed in the path
  def __init__(self, dim, files, capacity=1024, shuffleFile=True, shuffleRecord=True, epochs=None):
    self.dim = dim
    self.files = files
    self.shuffleFile = shuffleFile
    self.shuffleRecord = shuffleRecord
    self.epochs = epochs
    self.capacity = capacity
    self.nextfile = self._get_nextfile()
    self.nextline = self._read_line()
    self.lines = []
    
  def _get_nextfile(self):
    epoch, retidx = 0, 0
    
    filelist = list(self.files)
    if self.shuffleFile:
      random.shuffle(filelist)
      
    while True:
      if retidx>=len(filelist):
        epoch = epoch + 1 # 已经读完一轮
        retidx = 0
        if self.shuffleFile:
          random.shuffle(filelist)
        if self.epochs is None:
          epoch = 0
          continue
        else:
          if epoch<self.epochs:
            continue      
        break
      else:
        retidx=retidx+1
        yield filelist[retidx-1]
          
  def get_nextfile(self):
    return next(self.nextfile)
    
  def _read_line(self):
    while True:
      fname=self.get_nextfile()
      with open(fname, 'r') as f:
        for row in f:
          yield row.strip()

  def read_line(self):    
    return next(self.nextline)

  def read_batch(self, size=256):
    if self.capacity<size:
      self.capacity=size
    if len(self.lines)<self.capacity:
      self.lines = self.lines + list(islice(self.nextline, self.capacity-len(self.lines)))
    if self.shuffleRecord:
      random.shuffle(self.lines)
    if len(self.lines)<size:
      size=len(self.lines)
    ret=self.lines[:size]
    self.lines=self.lines[size:]
    return ret

  def read_labels_features_batch(self, size=256):
    labels = []
    features = []
    lines = self.read_batch(size)
    for line in lines:
      fields = line.strip().split()
      label = int(fields[0])
      feature = np.zeros(self.dim, dtype=float)

      for item in fields[1:]:
        index, value = item.split(':')
        idx = int(index)
        if idx < self.dim:
          feature[idx]=float(value)

      features.append(feature)
      labels.append(label)

    return labels, features, len(lines)

  def read_labels_features_map_batch(self, size=256):
    labels, features, lines_len = self.read_labels_features_batch(size)
    #labels = np.asarray(labels)
    return {'X':features, 'Y':labels, 'D': lines_len}

def simpleTest():
#  bread = LoadLibSvmData( 2048, ["data/20171031data.head.ridx", "data/20171101data.tail.ridx"], 
#   shuffleFile=False, shuffleRecord=False, capacity=100, epochs=1 )
    
  bread = LoadLibSvmData( dim=2048, files=["data/20171101data.tail.ridx"], 
    shuffleFile=False, shuffleRecord=False, capacity=100, epochs=1 )
    
#  print("%s" % )
#  for i in range(10):
#    labels, features, len_lines = bread.read_labels_features_batch(8)
#    print("%d %d %s" % (i, len_lines, str(features)) )

  for i in range(1):
    datamap = bread.read_labels_features_map_batch(8)
    print("%d %d %s %s" % (i, datamap['D'], str( datamap['Y']), type(datamap['Y'])) )
    print(str(datamap))

if __name__ == "__main__":
  #simpleQueueFetch()
  simpleTest()

