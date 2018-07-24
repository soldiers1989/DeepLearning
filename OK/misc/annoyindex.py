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
import random

t = AnnoyIndex(100)
klist={}
idx=0

#with open('all100000.txt', 'r', encoding="utf-8") as ins:
with open('result.txt', 'r', encoding="utf-8") as ins:
  for line in ins:
    fields=line.strip().split('\t')
    k=fields[0].replace(',', '')
    klist[idx]=k
    fields1=fields[1].split(' ')
    v=[float(x) for x in fields1[-100:]]
    t.add_item(idx, v)
    idx=idx+1

t.build(20) # 20 trees
t.save('vedio.ann')

gettop=600
with open('sampleresult', 'w', encoding="utf-8") as outs:
  for loopii in range(idx):
    qid=klist[loopii][:11]
    ids, dis = t.get_nns_by_item(loopii, gettop, include_distances=True)
    for ii in range(gettop):
      iinfo=list(klist[ids[ii]])
      iinfo.insert(11,'|')
      outs.write(str(qid)+'|'+str(ii)+'|'+str(ids[ii])+'|'+str(dis[ii])+'|'+"".join(iinfo))
      outs.write('\n')