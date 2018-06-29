idx=0
with open('D:\DeepLearning\data\model2.vec', 'r', encoding="utf-8") as inf:
  with open('D:\DeepLearning\data\model2.vec.proc', 'w', encoding="utf-8") as outf:
    for line in inf.readlines():
      if idx>0 :
        a=line.strip().split(' ')
        if len(a)!=101: print(line.strip())
        outf.write(str(idx)+' ')
        outf.write(line)
      idx+=1