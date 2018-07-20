import os
os.getcwd()
with open('.'+os.sep+'data'+os.sep+'all100000.csv', 'w', encoding="utf-8") as outs:
  with open('.'+os.sep+'data'+os.sep+'all100000.txt', 'r', encoding="utf-8") as ins:
    for line in ins:
      fields=line.strip().split('\t')
      field0=fields[0].replace(',', '')
      outs.write(field0+',')
      fields1=fields[1].split(' ')
      outs.write(','.join(fields1[-100:]))
      outs.write('\n')
