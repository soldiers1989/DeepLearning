idx=0
with open('D:\\DeepLearning\\data\\pred.result.shuf', 'r', encoding="utf-8") as inf:
  with open('D:\\DeepLearning\\data\\pred.result.shuf.csv', 'w', encoding="utf8") as outf:
    for line in inf.readlines():
      line=line.strip('\n').strip(' ')
      fields=line.split(' ')
      emb=','.join(fields[-100:])
      tag=' '.join(fields[0:-100])
      print(line)
      print(tag)
      print(emb)
#      print(vid)
#      fields = fields[1].split(' ')
#      emb=','.join(fields[-101:])
#      tag=' '.join(fields[0:-101])
      outf.write(','.join((tag,emb)))
      outf.write('\n')
#      idx+=1
#      if idx>100: break