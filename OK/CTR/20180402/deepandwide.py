# -*- coding: utf-8 -*-
#deep wide embedding 输出embedding，了解sparse_column_with_keys和sparse_column_with_hash_bucket的编码方式

import os
import pandas as pd
#import urllib.request
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

# Categorical base columns.
gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"], combiner="sqrtn")
race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=[
  "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"], combiner="sqrtn")
#education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
education = tf.contrib.layers.sparse_column_with_keys(column_name="education", keys=[
  "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Prof-school", "Some-college", "Assoc-acdm", "Assoc-voc", "Bachelors", "Masters", "Doctorate" ],
  combiner="sqrtn")
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100, combiner="sqrtn")
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100, combiner="sqrtn")
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000, combiner="sqrtn")
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000, combiner="sqrtn")

# Continuous base columns.
age = tf.contrib.layers.real_valued_column("age")
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

wide_columns = [
  gender, native_country, education, occupation, workclass, relationship, age_buckets,
  tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
  tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4)),
  tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6))]

deep_columns = [
  tf.contrib.layers.embedding_column(workclass, dimension=8),
  tf.contrib.layers.embedding_column(education, dimension=8),
  tf.contrib.layers.embedding_column(gender, dimension=8),
  tf.contrib.layers.embedding_column(relationship, dimension=8),
  tf.contrib.layers.embedding_column(native_country, dimension=8),
  tf.contrib.layers.embedding_column(occupation, dimension=8),
  age, education_num, capital_gain, capital_loss, hours_per_week]

wide_columns = [
  gender]

deep_columns = [
  tf.contrib.layers.embedding_column(education, dimension=8)]

model_dir =  '.'+os.sep+'data'+os.sep+'d_w_model'
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50],
    fix_global_step_increment_bug=True)

# Define the column names for the data sets.
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "gender",
  "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

df_train = pd.read_csv(
    tf.gfile.Open('.'+os.sep+'data'+os.sep+'adult.data'),
    names=COLUMNS,
    skipinitialspace=True,
    engine="python",
    skiprows=1)

df_test = pd.read_csv(
    tf.gfile.Open('.'+os.sep+'data'+os.sep+'adult.test'),
    names=COLUMNS,
    skipinitialspace=True,
    engine="python",
    skiprows=1)

df_train[LABEL_COLUMN] = (df_train['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)

def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(list(continuous_cols.items()) +
                        list(categorical_cols.items()))
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label

def train_input_fn():
    return input_fn(df_train)

def eval_input_fn():
    return input_fn(df_test)

m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

# Import this function from wherever it will end up in the future
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.contrib.layers import create_feature_spec_for_parsing

# I step
feature_columns = wide_columns + deep_columns

# II step
feature_spec = create_feature_spec_for_parsing(feature_columns)

# III step
export_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)

# IV step
servable_model_dir = '.'+os.sep+'data'+os.sep+'serving_savemodel'
servable_model_path = m.export_savedmodel(servable_model_dir, export_input_fn)
servable_model_path

m.get_variable_value('dnn/input_from_feature_columns/education_embedding/weights')

print(m.get_variable_value('dnn/input_from_feature_columns/education_embedding/weights'))
for key,value in enumerate(education.lookup_config[1]): print(key,value)

# from tensorflow.contrib import lookup
# educationtable = lookup.index_table_from_tensor(mapping=tuple(education.lookup_config.keys),
#   default_value=education.lookup_config.default_value, dtype=education.dtype, name="lookup")
# educationlookup = educationtable.lookup(tf.convert_to_tensor(education.lookup_config[1], dtype=tf.string))
#
# sess=tf.Session()
# educationtable.init.run(session=sess)
# print(educationlookup.eval(session=sess))
# sess.close()
#
# traindata=train_input_fn()
# occupationLookupArguments=occupation._wide_embedding_lookup_arguments(traindata[0]['occupation'])
# dir(occupationLookupArguments)

# from tensorflow.python.ops import string_ops
# sparse_id_values = string_ops.string_to_hash_bucket_fast(tf.convert_to_tensor(('Prof-specialty', 'Adm-clerical', 'Other-service', 'Handlers-cleaners'), dtype=tf.string),
#                                                          occupation.bucket_size, name="olookup")
# sess=tf.Session()
# print(sparse_id_values.eval(session=sess))
# sess.close()