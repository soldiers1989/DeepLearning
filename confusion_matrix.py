import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score

df = pd.read_csv('data/ft.result.csv', header=None)

df.head(5)

df.columns = ['y','pred1', 'pred2', 'pred3']

labels = df['y'].drop_duplicates().values.tolist()

for la in labels:
  subdf = df.loc[ (df['y'] == la) | (df['pred1'] == la), ['y','pred1']]
  ytrue=[1 if ii==la else 0 for ii in subdf['y'].values.tolist() ]
  ypred=[1 if ii==la else 0 for ii in subdf['pred1'].values.tolist() ]
  acc=accuracy_score(ytrue, ypred)
  prec=precision_score(ytrue, ypred, average='binary')
  recall=recall_score(ytrue, ypred, average='binary')
  
  subdf = df.loc[ df['y'] == la, ['y','pred1'] ]
  cnt=len(subdf)
  subdf = df.loc[  (df['y'] == la) & (df['pred1'] != la ), ['y','pred1'] ]
  errcnt=len(subdf)
  
  
  subdf2=subdf.groupby('pred1').size().nlargest(3)
  subdf2=subdf2.reset_index()
  subdf2.columns = ['err','count']
  err=subdf2.values.tolist()
  
  print('%s;%d;%d;%s;%f;%f;%f'%(la, cnt, errcnt, str(err), acc, prec, recall))




