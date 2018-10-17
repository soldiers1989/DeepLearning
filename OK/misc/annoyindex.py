from annoy import AnnoyIndex
import numpy as np

t = AnnoyIndex(100)  # Length of item vector that will be indexed
mps={}
with open('mp.pred', 'r', encoding="utf-8") as inf:
  for ii, line in enumerate(inf.readlines()):
    fds=line.strip().split(' ')
    mpv=np.array([float(x) for x in fds[-100:]])
    fds=fds[0].strip().split('_')
    bizuin,msgid,idx=fds[:3]
    title=''.join(fds[3:]).replace(',','')
    t.add_item(ii, mpv)
    mps[ii]=(','.join([title, bizuin, msgid, idx]), mpv, int(bizuin), title)

t.build(20) # 20 trees
t.save('mppred.ann')

gettop=1000
with open('mp.knn.pred', 'w', encoding="utf-8") as outf:
  for jj in range(len(mps)):
    item=mps[jj]
    ids, dis = t.get_nns_by_vector(item[1], gettop, include_distances=True)
    cnt=0
    for ii in range(gettop):
      if ids[ii]==jj: continue
      if mps[ids[ii]][3]==item[3]: continue
      cnt+=1
      outf.write('1,%d,%s,%s,%f\n' % (cnt, item[0], mps[ids[ii]][0], dis[ii]))
      if cnt==5: break
    cnt=0
    for ii in range(gettop):
      if item[2]!=mps[ids[ii]][2]: continue
      if mps[ids[ii]][3]==item[3]: continue
      if ids[ii]==jj: continue
      cnt+=1
      outf.write('0,%d,%s,%s,%f\n' % (cnt, item[0], mps[ids[ii]][0], dis[ii]))
      if cnt==5: break



with open('query.txt', 'r', encoding="utf-8") as ins:
  with open('queryresult50000', 'w', encoding="utf-8") as outs:
    for line in ins:
      fields=line.strip().split(' ')
      k=fields[0]
      v=[float(x) for x in fields[-100:]]
      ids, dis = t.get_nns_by_vector(v, gettop, include_distances=True)
      for ii in range(gettop):
        outs.write(k+'|'+str(ii)+'|'+klist[ids[ii]]+'|'+str(dis[ii]))
        outs.write('\n')






gettop=50000
with open('query.txt', 'r', encoding="utf-8") as ins:
  with open('queryresult50000', 'w', encoding="utf-8") as outs:
    for line in ins:
      fields=line.strip().split(' ')
      k=fields[0]
      v=[float(x) for x in fields[-100:]]
      ids, dis = t.get_nns_by_vector(v, gettop, include_distances=True)
      for ii in range(gettop):
        outs.write(k+'|'+str(ii)+'|'+klist[ids[ii]]+'|'+str(dis[ii]))
        outs.write('\n')














from annoy import AnnoyIndex
import random

f = 40
t = AnnoyIndex(f)  # Length of item vector that will be indexed
for i in range(1000):
  v = [random.gauss(0, 1) for z in range(f)]
  t.add_item(i, v)

t.build(10) # 10 trees
t.save('test.ann')

u = AnnoyIndex(f)
u.load('test.ann') # super fast, will just mmap the file
print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors


from annoy import AnnoyIndex


t = AnnoyIndex(100)
klist={}
idx=0

#with open('all100000.txt', 'r', encoding="utf-8") as ins:
with open('toquery.txt', 'r', encoding="utf-8") as ins:
  for line in ins:
    fields=line.strip().split(' ')
    k=fields[0]
    klist[idx]=k
    v=[float(x) for x in fields[-100:]]
    t.add_item(idx, v)
    idx=idx+1

t.build(20) # 20 trees
t.save('vedio.ann')
print('idx = %d' % idx)

gettop=50000
with open('query.txt', 'r', encoding="utf-8") as ins:
  with open('queryresult50000', 'w', encoding="utf-8") as outs:
    for line in ins:
      fields=line.strip().split(' ')
      k=fields[0]
      v=[float(x) for x in fields[-100:]]
      ids, dis = t.get_nns_by_vector(v, gettop, include_distances=True)
      for ii in range(gettop):
        outs.write(k+'|'+str(ii)+'|'+klist[ids[ii]]+'|'+str(dis[ii]))
        outs.write('\n')


#with open('sampleresult', 'w', encoding="utf-8") as outs:
#  for loopii in range(idx):
#    qid=klist[loopii][:11]
#    ids, dis = t.get_nns_by_item(loopii, gettop, include_distances=True)
#    for ii in range(gettop):
#      iinfo=list(klist[ids[ii]])
#      iinfo.insert(11,'|')
#      outs.write(str(qid)+'|'+str(ii)+'|'+str(ids[ii])+'|'+str(dis[ii])+'|'+"".join(iinfo))
#      outs.write('\n')

import numpy as np

mps=[]
with open('data/mp.pred', 'r', encoding="utf-8") as inf:
  for line in inf.readlines():
    fds=line.strip().split(' ')
    mpv=np.array([float(x) for x in fds[-100:]])
    fds=fds[0].strip().split('_')
    bizuin,msgid,idx=fds[-3:]
    title=''.join(fds[:-3]).replace(',','')
    #print[title,bizuin,msgid,idx]
    mps.append((','.join([title,bizuin,msgid,idx]), mpv, int(bizuin)))

with open('data/mp.knn.pred', 'w', encoding="utf-8") as outf:
  for ii in range(len(mps)):
    minds=0.0
    minidx=-1
    sameminds=0.0
    sameminidx = -1
    for jj in range(len(mps)):
      if ii==jj:continue
      cdis=mps[ii][1].dot(mps[jj][1])
      if cdis>=minds: minds, minidx=cdis, jj
      if mps[ii][2]!=mps[jj][2]: continue
      cdis=mps[ii][1].dot(mps[jj][1])
      if cdis>=sameminds: sameminds, sameminidx=cdis, jj
    if minidx >= 0:
      outf.write('1,%s,%s,%f\n' % (mps[ii][0], mps[minidx][0], minds))
    if sameminidx >= 0:
      outf.write('0,%s,%s,%f\n' % (mps[ii][0], mps[sameminidx][0], sameminds))
