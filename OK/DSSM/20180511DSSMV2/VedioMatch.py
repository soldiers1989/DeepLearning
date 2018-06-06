#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import time

import TFBCUtils
import numpy as np
import tensorflow as tf
from VedioMatchInputAnsy import VedioMatchInputAnsy

param = {
  'inputpath': 'data/',
  'modelpath': 'model/',
  'dataset': ['part-00000', 'part-00000'],
  'testset': ['part-00000', 'part-00000'],
  'predset': [],

  'shuffle_file': True,
  'batch_size': 10,
  'batch_size_test': 10,
  'test_batch': 100,
  'save_batch': 500,
  'total_batch': 1000,
  'decay_steps': 1000,
  'keep_prob': 0.8,

  'emb_size': 128,
  'userattr_size': 706,
  'vedioattr_size': 2000,  # 8368
  'margin': 0.2
}

param2 = {
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp_mv/',
  'modelpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp_mv/model/',

  'dataset': ['train/part-00000', 'train/part-00001', 'train/part-00002', 'train/part-00003', 'train/part-00004',
              'train/part-00005', 'train/part-00006', 'train/part-00007', 'train/part-00008', 'train/part-00009',
              'train/part-00010', 'train/part-00011', 'train/part-00012', 'train/part-00013', 'train/part-00014',
              'train/part-00015', 'train/part-00016', 'train/part-00017', 'train/part-00018', 'train/part-00019'],
  'testset': ['test/part-00000', 'test/part-00001', 'test/part-00002', 'test/part-00003', 'test/part-00004',
              'test/part-00005', 'test/part-00006', 'test/part-00007', 'test/part-00008', 'test/part-00009',
              'test/part-00010', 'test/part-00011', 'test/part-00012', 'test/part-00013', 'test/part-00014',
              'test/part-00015', 'test/part-00016', 'test/part-00017', 'test/part-00018', 'test/part-00019'],
  'predset': [],

  'shuffle_file': True,
  'batch_size': 256,
  'batch_size_test': 10240,
  'test_batch': 500,
  'save_batch': 5000,
  'total_batch': 100000,
  'decay_steps': 1000,
  'keep_prob': 0.8,

  'emb_size': 128,
  'userattr_size': 706,
  'vedioattr_size': 8369,
  'margin': 0.2,
}

param3 = {
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp_mv/',
  'modelpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp_mv/model/',

  'dataset': ['train_ran/part-00000', 'train_ran/part-00001', 'train_ran/part-00002', 'train_ran/part-00003',
              'train_ran/part-00004',
              'train_ran/part-00005', 'train_ran/part-00006', 'train_ran/part-00007', 'train_ran/part-00008',
              'train_ran/part-00009',
              'train_ran/part-00010', 'train_ran/part-00011', 'train_ran/part-00012', 'train_ran/part-00013',
              'train_ran/part-00014',
              'train_ran/part-00015', 'train_ran/part-00016', 'train_ran/part-00017', 'train_ran/part-00018',
              'train_ran/part-00019'],
  'testset': ['test_ran/part-00000', 'test_ran/part-00001', 'test_ran/part-00002', 'test_ran/part-00003',
                  'test_ran/part-00004',
                  'test_ran/part-00005', 'test_ran/part-00006', 'test_ran/part-00007', 'test_ran/part-00008',
                  'test_ran/part-00009',
                  'test_ran/part-00010', 'test_ran/part-00011', 'test_ran/part-00012', 'test_ran/part-00013',
                  'test_ran/part-00014',
                  'test_ran/part-00015', 'test_ran/part-00016', 'test_ran/part-00017', 'test_ran/part-00018',
                  'test_ran/part-00019'],
  'predset': [],

  'shuffle_file': True,
  'batch_size': 256,
  'batch_size_test': 10240,
  'test_batch': 500,
  'save_batch': 5000,
  'total_batch': 100000,
  'decay_steps': 1000,
  'keep_prob': 0.8,

  'emb_size': 128,
  'userattr_size': 706,
  'vedioattr_size': 8369,
  'margin': 0.2,
}


# param.update(param2)
param.update(param3)

def main():
  start_time = time.strftime('%m%d%Y%H%M', time.localtime(time.time()))
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  #  wxbiz_offline_db::daily_mid_wxsearch_kyk_video_log_share212_join_vedio_user_portrait_timeline_train_mv
  #  ds,uin,gender,age,grade,region_code,clickctype,clickclass_id_1,clickclass_id_2,clicktag,noclickctype,noclickclass_id_1,noclickclass_id_2,noclicktag
  #  t1vid,t1title,t1title0,t1class_id_1,t1class_id_2,t1tag
  #  t2vid,t2title,t2title1,t2class_id_1,t2class_id_2,t2tag
  # 用户
  # 用tf.nn.embedding_lookup gender, age, grade, region_code
  # 用tf.nn.embedding_lookup_sparse  clickctype
  # 用tf.nn.embedding_lookup_sparse  clickclass_id_1
  # 用tf.nn.embedding_lookup_sparse  clickclass_id_2
  # 用tf.nn.embedding_lookup_sparse  clicktag
  # 用tf.nn.embedding_lookup_sparse  noclickctype
  # 用tf.nn.embedding_lookup_sparse  noclickclass_id_1
  # 用tf.nn.embedding_lookup_sparse  noclickclass_id_2
  # 用tf.nn.embedding_lookup_sparse  noclicktag
  #
  # 视频
  # 用tf.nn.embedding_lookup t1title0,t1class_id_1,t1class_id_2
  # 用tf.nn.embedding_lookup_sparse t1tag

  ## define embeddings
  # 个人信息没有0
  embed_userattr = TFBCUtils.add_embedding(param['userattr_size'], param['emb_size'], '', 1,
                                           emb_name="embed_userattr")
  embed_vedioattr = TFBCUtils.add_embedding_with_zero(param['vedioattr_size'], param['emb_size'], '', 1,
                                                      emb_name="embed_vedioattr")

  ##----------------------------input
  with tf.name_scope('input') as scope:
    input_user_base = tf.placeholder(shape=[None, 4], dtype='int32', name='input_user_base')
    input_user_cctype = tf.sparse_placeholder(tf.int32, name='input_user_cctype')
    input_user_cclassid1 = tf.sparse_placeholder(tf.int32, name='input_user_cclassid1')
    input_user_cclassid2 = tf.sparse_placeholder(tf.int32, name='input_user_cclassid2')
    input_user_ctag = tf.sparse_placeholder(tf.int32, name='input_user_ctag')

    input_pos_vinfo = tf.placeholder(shape=[None, 3], dtype='int32', name='input_pos_vinfo')
    input_pos_ctag = tf.sparse_placeholder(tf.int32, name='input_pos_ctag')

    input_neg_vinfo = tf.placeholder(shape=[None, 3], dtype='int32', name='input_neg_vinfo')
    input_neg_ctag = tf.sparse_placeholder(tf.int32, name='input_neg_ctag')

  with tf.name_scope('param') as scope:
    keep_prob = tf.placeholder(dtype="float", name='keep_prob')
    global_step = tf.placeholder(dtype=np.int32, name="global_step")

  ##----------------------------embedding layer
  with tf.name_scope('embedding') as scope:
    with tf.device('/cpu:0'):
      embedding_user_base = tf.gather(embed_userattr, input_user_base, name='embedding_user_base')
      embedding_user_base = tf.reshape(embedding_user_base, [-1, 4 * param['emb_size']])
      embedding_user_cctype = tf.nn.embedding_lookup_sparse(embed_vedioattr, input_user_cctype, None, combiner='mean',
                                                            name='embedding_user_cctype')
      embedding_user_cclassid1 = tf.nn.embedding_lookup_sparse(embed_vedioattr, input_user_cclassid1, None,
                                                               combiner='mean', name='embedding_user_cclassid1')
      embedding_user_cclassid2 = tf.nn.embedding_lookup_sparse(embed_vedioattr, input_user_cclassid2, None,
                                                               combiner='mean', name='embedding_user_cclassid2')
      embedding_user_ctag = tf.nn.embedding_lookup_sparse(embed_vedioattr, input_user_ctag, None, combiner='mean',
                                                          name='embedding_user_ctag')

      embedding_pos_vinfo = tf.gather(embed_vedioattr, input_pos_vinfo, name='embedding_pos_vinfo')
      embedding_pos_vinfo = tf.reshape(embedding_pos_vinfo, [-1, 3 * param['emb_size']])
      embedding_pos_ctag = tf.nn.embedding_lookup_sparse(embed_vedioattr, input_pos_ctag, None, combiner='mean',
                                                         name='embedding_pos_ctag')

      embedding_neg_vinfo = tf.gather(embed_vedioattr, input_neg_vinfo, name='embedding_neg_vinfo')
      embedding_neg_vinfo = tf.reshape(embedding_neg_vinfo, [-1, 3 * param['emb_size']])
      embedding_neg_ctag = tf.nn.embedding_lookup_sparse(embed_vedioattr, input_neg_ctag, None, combiner='mean',
                                                         name='embedding_neg_ctag')

  ##----------------------------concat layer
  with tf.name_scope('concat') as scope:
    concat_pos = tf.concat([embedding_pos_vinfo, embedding_pos_ctag], 1)
    concat_neg = tf.concat([embedding_neg_vinfo, embedding_neg_ctag], 1)
    concat_user = tf.concat(
      [embedding_user_base, embedding_user_cctype, embedding_user_cclassid1, embedding_user_cclassid2,
       embedding_user_ctag], 1)

  ##----------------------------fc layer
  with tf.name_scope('fc') as scope:
    fc_user_w0, fc_user_b0 = TFBCUtils.create_w_b(8 * param['emb_size'], param['emb_size'] * 2,
                                                  w_name="fc_user_w0", b_name="fc_user_b0")
    fc_user_w1, fc_user_b1 = TFBCUtils.create_w_b(param['emb_size'] * 2, param['emb_size'], w_name="fc_user_w1",
                                                  b_name="fc_user_b1")

    fc_item_w0, fc_item_b0 = TFBCUtils.create_w_b(4 * param['emb_size'], param['emb_size'] * 2,
                                                  w_name="fc_item_w0", b_name="fc_item_b0")
    fc_item_w1, fc_item_b1 = TFBCUtils.create_w_b(param['emb_size'] * 2, param['emb_size'], w_name="fc_item_w1",
                                                  b_name="fc_item_b1")

    user_z0 = TFBCUtils.calculate_y(concat_user, fc_user_w0, fc_user_b0, keep_prob)
    user_z1 = TFBCUtils.calculate_y(user_z0, fc_user_w1, fc_user_b1, keep_prob)

    pos_z0 = TFBCUtils.calculate_y(concat_pos, fc_item_w0, fc_item_b0, keep_prob)
    pos_z1 = TFBCUtils.calculate_y(pos_z0, fc_item_w1, fc_item_b1, keep_prob)

    neg_z0 = TFBCUtils.calculate_y(concat_neg, fc_item_w0, fc_item_b0, keep_prob)
    neg_z1 = TFBCUtils.calculate_y(neg_z0, fc_item_w1, fc_item_b1, keep_prob)

    user_z1_n = tf.nn.l2_normalize(user_z1, 1)
    pos_z1_n = tf.nn.l2_normalize(pos_z1, 1)
    neg_z1_n = tf.nn.l2_normalize(neg_z1, 1)

  ##----------------------------loss layer
  with tf.name_scope('loss') as scope:
    loss = TFBCUtils.compute_triplet_loss(user_z1_n, pos_z1_n, neg_z1_n, param['margin'])
    learning_rate = tf.train.exponential_decay(0.002, global_step, param['decay_steps'], 0.98)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  # grads_and_vars = optimizer.compute_gradients(loss)
  #    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  with tf.name_scope('summary') as scope:
    # Summary.
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summaries.append(tf.summary.scalar("learning_rate", learning_rate))
    summary_op = tf.summary.merge(summaries)

  readdata = VedioMatchInputAnsy(param)
  debugdata = readdata.read_testdata_batch(size=5)
  debug_dict = {
    input_user_base: debugdata['user_base'],
    input_user_cctype: debugdata['user_cctype'],
    input_user_cclassid1: debugdata['user_cclassid1'],
    input_user_cclassid2: debugdata['user_cclassid2'],
    input_user_ctag: debugdata['user_ctag'],

    input_pos_vinfo: debugdata['pos_vinfo'],
    input_pos_ctag: debugdata['pos_ctag'],

    input_neg_vinfo: debugdata['neg_vinfo'],
    input_neg_ctag: debugdata['neg_ctag'],

    keep_prob: param['keep_prob'],
    global_step: 0
  }
  readdata.start_ansyc()

  with tf.Session(config=config) as session:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(param['modelpath'], graph=session.graph)
    summary_writer.add_graph(session.graph)
    step = 0

    for step in range(param['total_batch']):
      train_data = readdata.read_traindata_batch_ansyc()

      ## feed data to tf session
      feed_dict = {
        input_user_base: train_data['user_base'],
        input_user_cctype: train_data['user_cctype'],
        input_user_cclassid1: train_data['user_cclassid1'],
        input_user_cclassid2: train_data['user_cclassid2'],
        input_user_ctag: train_data['user_ctag'],

        input_pos_vinfo: train_data['pos_vinfo'],
        input_pos_ctag: train_data['pos_ctag'],

        input_neg_vinfo: train_data['neg_vinfo'],
        input_neg_ctag: train_data['neg_ctag'],

        keep_prob: param['keep_prob'],
        global_step: step
      }

      if (step > 0 and step % param['test_batch'] == 0):
        ## run optimizer
        _, l, s, lr = session.run([optimizer, loss, summary_op, learning_rate], feed_dict=feed_dict)
        print('[%s]\t[Train]\tIter:%d\tloss=%.6f\tlr=%.6f' % (
        time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), step, l, lr), end='\n')

        summary_writer.add_summary(s, step)

        lv, user_z1_out, pos_z1_out, neg_z1_out, pos_i, neg_i, user_i = \
          session.run([loss, user_z1_n, pos_z1_n, neg_z1_n, concat_pos, concat_neg, concat_user], \
                      feed_dict=debug_dict)
        TFBCUtils.print_model_data(lv, user_z1_out, pos_z1_out, neg_z1_out, pos_i, neg_i, user_i)

        test_data = readdata.read_testdata_batch_ansyc()
        feed_dict = {
          input_user_base: test_data['user_base'],
          input_user_cctype: test_data['user_cctype'],
          input_user_cclassid1: test_data['user_cclassid1'],
          input_user_cclassid2: test_data['user_cclassid2'],
          input_user_ctag: test_data['user_ctag'],

          input_pos_vinfo: test_data['pos_vinfo'],
          input_pos_ctag: test_data['pos_ctag'],

          input_neg_vinfo: test_data['neg_vinfo'],
          input_neg_ctag: test_data['neg_ctag'],

          keep_prob: 1.0,
          global_step: step
        }

        ## get acc
        lv, user_z1_out, pos_z1_out, neg_z1_out = \
          session.run([loss, user_z1_n, pos_z1_n, neg_z1_n], feed_dict=feed_dict)

        acc, prec, rec, auc = TFBCUtils.get_accprecrecauc(user_z1_out, pos_z1_out, neg_z1_out, param['margin'])
        print('[Test]\tIter:%d\tloss=%.6f\taccuracy=%.6f\tprecision=%.6f\trecall=%.6f\tauc=%.6f' % (
        step, lv, acc, prec, rec, auc), end='\n')
        print("-----------------------------------------")

        summary = tf.Summary(value=[
          tf.Summary.Value(tag="train_loss", simple_value=l.item()),
          tf.Summary.Value(tag="test_loss", simple_value=lv.item()),
          tf.Summary.Value(tag="test_acc", simple_value=acc),
          tf.Summary.Value(tag="test_auc", simple_value=auc),
        ])

        summary_writer.add_summary(summary, step)
      else:
        ## run optimizer
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

      if (step > 0 and step % param['save_batch'] == 0):
        model_name = "dnn-model-" + start_time + '-' + str(step)
        if not os.path.exists(param['modelpath']):
          os.mkdir(param['modelpath'])
        saver.save(session, os.path.join(param['modelpath'], model_name))

    summary_writer.close()

  readdata.stop_and_wait_ansyc()


if __name__ == "__main__":
  main()
