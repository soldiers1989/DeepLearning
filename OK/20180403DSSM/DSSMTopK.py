# coding: utf-8

import sys
import numpy as np
import math
from MPInputAnsy import TxtFilesRandomReader

Py3 = sys.version_info[0] == 3
if Py3:
  from queue import PriorityQueue
else:
  from Queue import PriorityQueue

param = {
  'inputpath' : 'data/',
  'predset'   : ['mp.pred.bak'], # ['mp.pred.bak'],
  'vector'    : '-1.0#-1.0#-0.99979836#-1.0#-1.0#1.0#1.0#-1.0#-1.0#-1.0#0.99701345#-0.9999949#-0.9998324#1.0#-1.0#0.99999666#-0.9707126#-0.99997896#0.9999985#1.0#-0.99985754#-1.0#-1.0#-1.0#-1.0#1.0#-0.9998259#-0.99996555#1.0#-1.0#0.99976856#1.0#-0.99887997#0.9992179#-1.0#1.0#0.99998695#-1.0#-0.9924827#-1.0#-1.0#-0.99759424#-0.99998844#1.0#0.99999994#-1.0#-0.9999997#0.9875487#1.0#-0.2777283#-1.0#-0.9999994#-0.99975514#1.0#-0.012020128#-0.9999921#0.44607463#-0.98461366#1.0#-1.0#0.9938352#0.92664313#0.9998989#0.99999154#-1.0#-1.0#0.9999998#1.0#-1.0#0.9957686#-1.0#0.99999297#0.99999994#-1.0#0.9894841#-1.0#1.0#1.0#-0.99998754#-0.99999976#-0.9996472#-0.9999995#0.9999985#-0.98746157#-0.9999987#1.0#-0.9999557#-1.0#-1.0#-0.9999998#-1.0#1.0#1.0#-1.0#0.9242719#-1.0#1.0#0.99999934#0.998326#-1.0#1.0#-0.99999964#0.049065143#1.0#0.13535279#1.0#1.0#1.0#-0.99999976#-1.0#1.0#-0.999999#0.9734229#0.9999786#0.9999995#1.0#-1.0#0.9999917#-0.98444337#0.99928087#1.0#-1.0#-1.0#-1.0#-1.0#0.9999146#-1.0#-0.99998057',
  'top'       : 20
}

def cosine_similarity(v1, v2):
  "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
  sumxx, sumxy, sumyy = 0, 0, 0
  for i in range(len(v1)):
    x = v1[i]; y = v2[i]
    sumxx += x*x
    sumyy += y*y
    sumxy += x*y
  return sumxy/math.sqrt(sumxx*sumyy)

def getVector(strinput):
  fields=strinput.split('#')
  return [float(x) for x in fields]

if __name__ == '__main__':
  pQueue = PriorityQueue()
  
  vleft = getVector(param['vector'])
  
  predset = [ param['inputpath'] + item for item in param['predset'] ]
  preddata = TxtFilesRandomReader(files=predset, shuffleFile=False, shuffleRecord=False, epochs=1)
  
  lines = preddata.read_batch(size=5)
  while len(lines)>0:
    for line in lines:
      fields = line.strip().split('\t')
      if len(fields) != 2: continue
      vright = getVector(fields[1])
      sim = cosine_similarity(vleft, vright)
      pQueue.put((sim, line))
      topop = pQueue.qsize() - param['top']
      while topop>0:
        pQueue.get()
        topop=topop-1
  
    lines = preddata.read_batch(size=5)

  print('*'*20)
  topop = pQueue.qsize()
  while topop>0:
    item=pQueue.get()
    print('%f %s' % item)
