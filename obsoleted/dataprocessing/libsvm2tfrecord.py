#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
import sys, os
import tensorflow as tf
import argparse
flags = tf.app.flags
FLAGS = flags.FLAGS
_float_feature = lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v))
_int_feature = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def out_file(f):
  if f.endswith(".csv") or f.endswith(".txt"):
    return f.replace(".csv", ".tfrecord").replace(".txt", ".tfrecord")
  else:
    return f + ".tfrecord"

def libsvm2tfrecord(infile):
  outfile=out_file(infile)
  writer = tf.python_io.TFRecordWriter(outfile)
  for line in open(infile):
    fields = line.rstrip().split()
    label = int(fields[0])
      
    indexes, values = [], []

    for item in fields[1:]:
      index, value = item.split(':')
      indexes.append(int(index))
      values.append(float(value))

    example = tf.train.Example(features=tf.train.Features(feature={
      'label': _int_feature([label]),
      'index': _int_feature(indexes),
      'value': _float_feature(values)
    }))
    writer.write(example.SerializeToString())
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run libsvm to tfrecord.")
  parser.add_argument('--path', default='data/', required=True,
            help='Input data path.')
  parser.add_argument('--inputfiles', nargs='?', default='', required=True,
            help='Choose inputfiles, seperated by #.')
  args = parser.parse_args()
  
  files = args.inputfiles.split('#')
  for f in files:
    fname = args.path+'/'+f
    libsvm2tfrecord(fname)
