# coding: utf-8

import math
import sys

from MPInputAnsyVedio import TxtFilesRandomReader

Py3 = sys.version_info[0] == 3
if Py3:
  from queue import PriorityQueue
else:
  from Queue import PriorityQueue

param = {
  'inputpath' : 'data/',
  'predset'   : ['baidu.pred'], # ['mp.pred.bak'],
  'vector'    : '0.8929579#0.79427713#-0.84723324#0.9492298#-0.7937324#0.6364886#-0.81177413#0.6222893#-0.26429668#0.6363772#-0.5980054#0.15300731#0.7405207#0.7880381#0.7236987#-0.6595422#0.009318216#0.7547381#-0.88029724#-0.84537965#0.63022375#0.6958265#-0.84398973#0.50388217#-0.67082673#-0.68813527#0.72877467#-0.8427424#0.12283948#-0.8046819#-0.7336247#-0.88780904#0.17052595#-0.77914965#0.8835975#0.6966787#0.6209818#0.85156876#-0.7017073#0.89421624#-0.8235925#-0.65330315#0.9002207#0.6983923#0.7457568#0.64927256#0.34535185#-0.37334242#0.82675153#0.67968714#-0.714329#-0.7931291#0.63731253#0.8242302#-0.87746686#0.91399765#-0.13916875#-0.7210654#-0.5053742#-0.75823915#-0.8886578#-0.7738352#0.66519666#-0.9278239#-0.7445612#0.70204854#-0.7653221#-0.63277936#-0.7957371#0.6907899#0.6946426#-0.72005093#-0.7750614#-0.64455986#-0.8557586#0.73261905#-0.7336433#-0.735198#-0.74548244#0.7485671#0.14400497#-0.71596766#-0.88880634#0.816394#0.67422676#-0.85644597#-0.68526685#-0.83224225#-0.7382113#0.8186215#0.7673648#0.5976677#0.62225735#-0.710544#-0.6260291#-0.7079823#-0.72200036#0.7886571#0.9360182#0.8140873#0.8293141#-0.7345596#0.67861605#0.6625347#0.62086#-0.68327194#-0.76867384#0.8141341#-0.48194376#-0.24910969#-0.7900697#0.71457165#0.7012247#-0.80265236#0.58020175#-0.8115134#-0.79950833#-0.7209815#-0.7796326#-0.77238935#0.7057121#-0.82950705#0.6886256#0.7992563#0.801824#-0.83538395#0.80679286#0.62283385',
  'top'       : 50
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
