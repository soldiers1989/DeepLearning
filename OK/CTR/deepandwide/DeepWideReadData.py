#-*- coding: utf-8 -*-

import os
import argparse
import pickle

#--以下是稀疏的
#0 agebucket
#1 genderbucket
#2 gradebucket
#3 userfulllongreadldamap, 
#4 userfullshortreadldamap, 
#5 usernotclickldamap, 
#6 msgreadldamap
#7 userlongreadtagmap, 
#8 usershortreadtagmap, 
#9 usernotclicktagmap, 
#10 msgreadtagmap
#11 bizuin
#12 regioncode
#--以下是稠密的
#13 user_long_readpv
#14 user_short_readpv
#15 msgfulllongreadinnerprod
#16 msgfullshortreadinnerprod
#17 msglongreadtaginnerprod
#18 msgshortreadtaginnerprod
#19 msgnotclickinnerprod
#20 msgnotclicktaginnerprod
#21 embed1
#22 embed2
# [10, 3, 3, 1600, 1600, 1600, 1600, 100000, 100000, 100000, 100000, 10000, 5000, 1, 1, 1, 1, 1, 1, 1, 1, 128, 100]

class DeepWideReadData(object):
  def __init__(self, inputargs):
    self.args = inputargs
    self.process_conf(eval(
'''
[
  (0,'agebucket',1,10),
  (1,'genderbucket',1,3),
  (2,'gradebucket',1,3),
  (3,'userfulllongreadldamap',1,1600),
  (4,'userfullshortreadldamap',1,1600),
  (5,'usernotclickldamap',1,1600),
  (6,'msgreadldamap',1,1600),
  (7,'userlongreadtagmap',1,100000),
  (8,'usershortreadtagmap',1,100000),
  (9,'usernotclicktagmap',1,100000),
  (10,'msgreadtagmap',1,100000),
  (11,'bizuin',1,10000),
  (12,'regioncode',1,5000),
  (13,'user_long_readpv',0,1),
  (14,'user_short_readpv',0,1),
  (15,'msgfulllongreadinnerprod',0,1),
  (16,'msgfullshortreadinnerprod',0,1),
  (17,'msglongreadtaginnerprod',0,1),
  (18,'msgshortreadtaginnerprod',0,1),
  (19,'msgnotclickinnerprod',0,1),
  (20,'msgnotclicktaginnerprod',0,1),
  (21,'embed1',0,128),
  (22,'embed2',0,100) 
]
'''))

  def process_conf(self, conf):
    begin=1
    self.feature_conf=[]
    for item in conf:
      self.feature_conf.append( item + (begin, begin+item[3]-1) )
      begin += item[3]
    self.feature_num=len(self.feature_conf)

  def parse_data(self, line):
    fields = line.strip().split()
    label = int(fields[0])

    feaures = [ (item.split(':')) for item in fields[1:] ]
    feaures = sorted( filter(lambda x:x[1]!=0, [ (int(item[0]),float(item[1])) for item in feaures ] ), key = lambda a:a[0])
    return label, feaures

  def print_data(self, line):
    label, feaures = self.parse_data(line)
    print(feaures)
    
    fidx=0
    for item in feaures:
      while fidx<=self.feature_num:
        if self.feature_conf[fidx][4]<=item[0] and item[0]<=self.feature_conf[fidx][5]:
          print(" Name: "+self.feature_conf[fidx][1]+" Value:"+str(item))
          break
        else:
          fidx+=1
      print(" Unknown: "+self.feature_conf[fidx][1]+" Value:"+str(item))
          
#  def ftrl_feature(self):
  	
    
    
    
    
    
    
    
#    importance = sorted(importance.items(), key = lambda a:a[1], reverse = True)
#
#      for item in fields[1:]:
#        index, value = item.split(':')
#        idx = int(index)
#        if idx < self.dim:
#          feature[idx]=float(value)

#    self.inputpath = args.inputpath     #输入目录
#    self.outputpath = args.outputpath   #输出目录
#    self.inputfiles = args.inputfiles   #待处理文件列表
#    self.postfix = args.postfix         #处理结果后缀
#    self.remove_embedding = args.remove_embedding #删除embedding
#
#    self.idmap = {}
#    self.maxidx = 0
#    self.appendidx = args.appendidx
#    
#    self.skipidxline = 0
#    self.skipidxfield = 0
#    self.skipembeddingfield = 0
#    self.totalline = 0
#    self.totalfield = 0
#
#    self.idxfile = args.idxfile
#    self.idxfilename = self.inputpath + '/' + self.idxfile
#    if args.loadidxfile:
#      if os.path.isfile(self.idxfilename):
#        self.loadidmap()
#        print("index file " + self.idxfilename + " Loaded")
#      else:
#        print("index file "+self.idxfilename+" NOT EXISTED")
#    else:
#      self.appendidx = True             #如果不加载index文件
#    
#    print("self.appendidx: "+str(self.appendidx))


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
#  parser.add_argument('--outputpath', default='data/', required=True,
#            help='Output data path.')
#  parser.add_argument('--inputfiles', nargs='?', default='', required=True,
#            help='Choose inputfiles, seperated by #.')
#  parser.add_argument('--postfix', default='ridx',
#            help='Reindex postfix.')
#  parser.add_argument('--idxfile', default='ridxfile',
#            help='index mapping file.')
#  parser.add_argument('--loadidxfile', type=str2bool, default=False,
#            help='Loading existed index.')
#  parser.add_argument('--appendidx', type=str2bool, default=False,
#            help='Add new index. If idxfile NOT EXISTED, it will set to True')
#  parser.add_argument('--remove_embedding', type=str2bool, default=False,
#            help='Remove Embedding Feature')
  args = parser.parse_args()
  readdata = DeepWideReadData(args)
  print(readdata.feature_conf)
  readdata.print_data('0 5:1 13:1 16:1 107:0.0109 115:0.0639 134:0.0270 262:0.0103 289:0.0193 453:0.0133 526:0.0136 533:0.0171 561:0.0177 565:0.0118 590:0.0447 637:0.0154 658:0.0222 712:0.0194 722:0.0174 772:0.0276 794:0.0122 985:0.0151 1013:0.0450 1114:0.0310 1126:0.0266 1262:0.0200 1296:0.0119 1316:0.0102 1446:0.0317 1492:0.0422 1506:0.0105 1634:0.0137 1641:0.0210 1734:0.0227 1808:0.0109 1903:0.0111 1917:0.0237 2138:0.0106 2139:0.0142 2144:0.0114 2190:0.0327 2258:0.0124 2355:0.0234 2455:0.0116 2479:0.0464 2494:0.0173 2498:0.0167 2613:0.0848 2714:0.0164 2731:0.0150 2736:0.0177 2748:0.0179 2952:0.0158 3092:0.0143 3124:0.0207 3133:0.0229 3232:0.0470 3234:0.1189 3350:0.0124 3403:0.0151 3421:0.0167 3518:0.0189 3531:0.0148 3578:0.0105 3601:0.0151 3636:0.0149 3691:0.0334 3887:0.0155 3948:0.0138 3955:0.0108 3990:0.0578 4098:0.1322 4335:0.0322 4348:0.1253 4645:0.0173 4683:0.0825 4716:0.0229 5867:0.0896 5948:0.6628 6084:0.0420 407798:1 418128:1 421417:0.1430 421418:0.1009 421419:0.0000 421420:0.1252 421421:0.0000 421422:0.0000 421423:0.4816 421424:0.0000 421425:0.0 421426:0.0 421427:0.0 421428:0.0 421429:0.0 421430:0.0 421431:0.0 421432:0.0 421433:0.0 421434:0.0 421435:0.0 421436:0.0 421437:0.0 421438:0.0 421439:0.0 421440:0.0 421441:0.0 421442:0.0 421443:0.0 421444:0.0 421445:0.0 421446:0.0 421447:0.0 421448:0.0 421449:0.0 421450:0.0 421451:0.0 421452:0.0 421453:0.0 421454:0.0 421455:0.0 421456:0.0 421457:0.0 421458:0.0 421459:0.0 421460:0.0 421461:0.0 421462:0.0 421463:0.0 421464:0.0 421465:0.0 421466:0.0 421467:0.0 421468:0.0 421469:0.0 421470:0.0 421471:0.0 421472:0.0 421473:0.0 421474:0.0 421475:0.0 421476:0.0 421477:0.0 421478:0.0 421479:0.0 421480:0.0 421481:0.0 421482:0.0 421483:0.0 421484:0.0 421485:0.0 421486:0.0 421487:0.0 421488:0.0 421489:0.0 421490:0.0 421491:0.0 421492:0.0 421493:0.0 421494:0.0 421495:0.0 421496:0.0 421497:0.0 421498:0.0 421499:0.0 421500:0.0 421501:0.0 421502:0.0 421503:0.0 421504:0.0 421505:0.0 421506:0.0 421507:0.0 421508:0.0 421509:0.0 421510:0.0 421511:0.0 421512:0.0 421513:0.0 421514:0.0 421515:0.0 421516:0.0 421517:0.0 421518:0.0 421519:0.0 421520:0.0 421521:0.0 421522:0.0 421523:0.0 421524:0.0 421525:0.0 421526:0.0 421527:0.0 421528:0.0 421529:0.0 421530:0.0 421531:0.0 421532:0.0 421533:0.0 421534:0.0 421535:0.0 421536:0.0 421537:0.0 421538:0.0 421539:0.0 421540:0.0 421541:0.0 421542:0.0 421543:0.0 421544:0.0 421545:0.0 421546:0.0 421547:0.0 421548:0.0 421549:0.0 421550:0.0 421551:0.0 421552:0.0 421553:0.0088 421554:0.0225 421555:0.1119 421556:-0.1752 421557:-0.1107 421558:0.0432 421559:0.0237 421560:-0.1678 421561:0.0003 421562:0.0308 421563:-0.0935 421564:-0.0022 421565:-0.0529 421566:-0.0007 421567:-0.0311 421568:-0.0456 421569:0.1593 421570:-0.01 421571:0.0199 421572:0.0625 421573:-0.0461 421574:0.0283 421575:0.0014 421576:-0.0659 421577:-0.0676 421578:-0.2543 421579:-0.0065 421580:-0.2427 421581:0.0241 421582:-0.0114 421583:-0.0497 421584:0.0359 421585:0.0734 421586:-0.0051 421587:0.025 421588:-0.0289 421589:0.0235 421590:-0.1095 421591:0.1251 421592:0.03 421593:0.1832 421594:-0.1141 421595:0.0527 421596:0.0664 421597:0.1162 421598:0.0707 421599:-0.0168 421600:-0.1876 421601:-0.0206 421602:0.0479 421603:-0.1697 421604:0.0143 421605:-0.0078 421606:0.0376 421607:0.0216 421608:-0.1257 421609:0.0468 421610:-0.0078 421611:-0.025 421612:0.054 421613:-0.0104 421614:-0.0605 421615:0.0011 421616:-0.1121 421617:-0.2165 421618:0.0522 421619:-0.0234 421620:0.02 421621:-0.0455 421622:-0.0363 421623:-0.1341 421624:0.0753 421625:-0.0295 421626:-0.1534 421627:-0.0075 421628:0.2945 421629:0.0468 421630:-0.5243 421631:0.2086 421632:0.3993 421633:0.0217 421634:-0.1039 421635:0.0056 421636:0.0072 421637:0.148 421638:-0.0585 421639:0.0159 421640:-0.2327 421641:0.0444 421642:-0.0581 421643:-0.0204 421644:0.0228 421645:-0.0229 421646:-0.0674 421647:0.0335 421648:-0.0686 421649:-0.0953 421650:-0.1477 421651:0.0506 421652:-0.0875')