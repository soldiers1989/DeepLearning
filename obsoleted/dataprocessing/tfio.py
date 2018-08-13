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


def simpleQueue():
  # filename_queue = tf.train.string_input_producer(["data/20171101data.tail"], num_epochs=3)
  filename_queue = tf.train.string_input_producer(["data/20171101data.tail", "data/20171100data.tail"], shuffle=True,
                                                  num_epochs=3)
  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)

  with tf.Session() as sess:
    sess.run(tf.initialize_local_variables())
    tf.train.start_queue_runners()
    num_examples = 0
    try:
      while True:
        s_key, s_value = sess.run([key, value])
        print(s_key, s_value)
      num_examples += 1
    except tf.errors.OutOfRangeError:  # 一直读到结束
      print("There are", num_examples, "examples")


def simpleQueueFetch():
  # filename_queue = tf.train.string_input_producer(["data/20171101data.tail"], num_epochs=3)
  filename_queue = tf.train.string_input_producer(["data/20171101data.tail", "data/20171100data.tail"], shuffle=True,
                                                  num_epochs=3)
  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)

  with tf.Session() as sess:
    sess.run(tf.initialize_local_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(25):
      s_key, s_value = sess.run([key, value])
      print(s_key, s_value)

    coord.request_stop()
    coord.join(threads)


class BatchReadData(object):
  # Three files are needed in the path
  def __init__(self, files, shuffle, epochs=None):
    self.files = files
    self.shuffle = shuffle
    self.epochs = epochs
    self.filename_queue = tf.train.string_input_producer(files, shuffle=shuffle, num_epochs=epochs)
    self.reader = tf.TextLineReader()

  def readBatch(self, batch_size=5):
    MIN_AFTER_DEQUEUE = 1000
    key, value = self.reader.read(self.filename_queue)
    key, value = tf.train.shuffle_batch(
      [key, value],
      batch_size=batch_size,
      capacity=MIN_AFTER_DEQUEUE + 3 * batch_size,
      # Ensures a minimum amount of shuffling of examples.
      min_after_dequeue=MIN_AFTER_DEQUEUE)
    return key, value


def simpleQueueBatchFetch():
  with tf.Session() as sess:
    sess.run(tf.initialize_local_variables())

    bread = BatchReadData(["data/20171101data.tail", "data/20171100data.tail"], True)
    key, value = bread.readBatch(2)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(5):
      s_key, s_value = sess.run([key, value])
      print(s_key, s_value)

    coord.request_stop()
    coord.join(threads)


#if __name__ == "__main__":
  # simpleQueueFetch()
#  simpleQueueBatchFetch()