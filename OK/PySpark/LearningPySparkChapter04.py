import sys
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as fn
import pyspark.sql.types as typ

#conf = (SparkConf()
#         .setMaster("local")
#         .setAppName("LearningPySparkChapter04")
#         .set("spark.executor.memory", "1g"))
#sc = SparkContext(conf=conf)

sc = SparkContext("local", "LearningPySparkChapter03")
spark = SparkSession.builder.master("local").appName("Python Spark SQL basic example").config("spark.sql.warehouse.dir","file:///").getOrCreate()

#df = spark.createDataFrame([
#        (1, 144.5, 5.9, 33, 'M'),
#        (2, 167.2, 5.4, 45, 'M'),
#        (3, 124.1, 5.2, 23, 'F'),
#        (4, 144.5, 5.9, 33, 'M'),
#        (5, 133.2, 5.7, 54, 'F'),
#        (3, 124.1, 5.2, 23, 'F'),
#        (5, 129.2, 5.3, 42, 'M'),
#    ], ['id', 'weight', 'height', 'age', 'gender'])
#
#print('Count of rows: {0}'.format(df.count()))
#print('Count of distinct rows: {0}'.format(df.distinct().count()))
#
#df = df.dropDuplicates()
#print('*'*20+'dropDuplicates 1')
#df.show()
#
#print('Count of ids: {0}'.format(df.count()))
#print('Count of distinct ids: {0}'.format(df.select([c for c in df.columns if c != 'id']).distinct().count()))
#
#df = df.dropDuplicates(subset=[c for c in df.columns if c != 'id'])
#print('*'*20+'dropDuplicates 2')
#df.show()
#
#import pyspark.sql.functions as fn
#
#print('*'*20+'show 3')
#df.agg(
#    fn.count('id').alias('count'),
#    fn.countDistinct('id').alias('distinct')
#).show()
#
#print('*'*20+'show 4')
#df.withColumn('new_id', fn.monotonically_increasing_id()).show()

#####################  Missing observations
#df_miss = spark.createDataFrame([
#        (1, 143.5, 5.6, 28,   'M',  100000),
#        (2, 167.2, 5.4, 45,   'M',  None),
#        (3, None , 5.2, None, None, None),
#        (4, 144.5, 5.9, 33,   'M',  None),
#        (5, 133.2, 5.7, 54,   'F',  None),
#        (6, 124.1, 5.2, None, 'F',  None),
#        (7, 129.2, 5.3, 42,   'M',  76000),
#    ], ['id', 'weight', 'height', 'age', 'gender', 'income'])
#
#df_miss.rdd.map(
#    lambda row: (row['id'], sum([c == None for c in row]))
#).collect()
#
#df_miss.where('id == 3').show()
#
#df_miss.agg(*[
#    (1 - (fn.count(c) / fn.count('*'))).alias(c + '_missing')
#    for c in df_miss.columns
#]).show()  # by columns
#
#df_miss_no_income = df_miss.select([c for c in df_miss.columns if c != 'income'])
#df_miss_no_income.show()  # remove column
#
#df_miss_no_income.dropna(thresh=3).show()  # dropna
#
#means = df_miss_no_income.agg(
#    *[fn.mean(c).alias(c) for c in df_miss_no_income.columns if c != 'gender']
#).toPandas().to_dict('records')[0]  # mean and fillna
#
#means['gender'] = 'missing'
#
#df_miss_no_income.fillna(means).show()


######################  Outliers
#df_outliers = spark.createDataFrame([
#        (1, 143.5, 5.3, 28),
#        (2, 154.2, 5.5, 45),
#        (3, 342.3, 5.1, 99),
#        (4, 144.5, 5.5, 33),
#        (5, 133.2, 5.4, 54),
#        (6, 124.1, 5.1, 21),
#        (7, 129.2, 5.3, 42),
#    ], ['id', 'weight', 'height', 'age'])
#
#cols = ['weight', 'height', 'age']
#bounds = {}
#
#for col in cols:
#    quantiles = df_outliers.approxQuantile(col, [0.25, 0.75], 0.05)
#    IQR = quantiles[1] - quantiles[0]
#    bounds[col] = [quantiles[0] - 1.5 * IQR, quantiles[1] + 1.5 * IQR]  #approxQuantile
#
#print(bounds)
#
#outliers = df_outliers.select(*['id'] + [
#    (
#        (df_outliers[c] < bounds[c][0]) | 
#        (df_outliers[c] > bounds[c][1])
#    ).alias(c + '_o') for c in cols
#])
#outliers.show()
#
##We have two outliers in the weight feature and two in the age feature.
#df_outliers = df_outliers.join(outliers, on='id')
#df_outliers.filter('weight_o').select('id', 'weight').show()
#df_outliers.filter('age_o').select('id', 'age').show()


##################### Understand your data

#https://stackoverflow.com/questions/37354583/error-with-first-step-in-pyspark
#My experience in these situations is that an executor ran into a resource paucity, and was killed. Typically a shortage of memory.
fraud = sc.textFile('data/ccFraud.csv.gz')
header = fraud.first()

fraud = fraud.filter(lambda row: row != header).map(lambda row: [int(elem) for elem in row.split(',')])

fields = [
    *[
        typ.StructField(h[1:-1], typ.IntegerType(), True)
        for h in header.split(',')
    ]
]

schema = typ.StructType(fields)
print(schema)

fraud_df = spark.createDataFrame(fraud, schema)

fraud_df.printSchema()

fraud_df.groupby('gender').count().show()  # groupby

numerical = ['balance', 'numTrans', 'numIntlTrans']

#For the truly numerical features we can use the .describe() method.
desc = fraud_df.describe(numerical)
desc.show()

fraud_df.agg({'balance': 'skewness'}).show()
  
print(fraud_df.corr('balance', 'numTrans'))     

n_numerical = len(numerical)

corr = []

for i in range(0, n_numerical):
    temp = [None] * i
    
    for j in range(i, n_numerical):
        temp.append(fraud_df.corr(numerical[i], numerical[j]))
    corr.append(temp)
    
print(corr)

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import bokeh.charts as chrt
from bokeh.io import output_notebook

hists = fraud_df.select('balance').rdd.flatMap(lambda row: row).histogram(20)
data = {
    'bins': hists[0][:-1],
    'freq': hists[1]
}

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(1, 1, 1)
ax.bar(data['bins'], data['freq'], width=2000)
ax.set_title('Histogram of \'balance\'')

plt.savefig('B05793_05_22.png', dpi=300)

b_hist = chrt.Bar(data, values='freq', label='bins', title='Histogram of \'balance\'')
chrt.show(b_hist)

data_driver = {'obs': fraud_df.select('balance').rdd.flatMap(lambda row: row).collect()}
  
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(1, 1, 1)

ax.hist(data_driver['obs'], bins=20)
ax.set_title('Histogram of \'balance\' using .hist()')


plt.savefig('B05793_05_24.png', dpi=300)

b_hist_driver = chrt.Histogram(data_driver, values='obs', title='Histogram of \'balance\' using .Histogram()', bins=20)
chrt.show(b_hist_driver)

data_sample = fraud_df.sampleBy('gender', {1: 0.0002, 2: 0.0002}).select(numerical)
  
data_multi = dict([
    (elem, data_sample.select(elem).rdd.flatMap(lambda row: row).collect()) 
    for elem in numerical
])

sctr = chrt.Scatter(data_multi, x='balance', y='numTrans')

chrt.show(sctr)

sc.stop()
