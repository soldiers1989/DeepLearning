#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import time

import VedioMatchUtils
import numpy as np
import tensorflow as tf
from VedioClassifyInputAnsy import VedioDSSMInputAnsy

param = {
  'inputpath': 'data/',
  'modelpath': 'model/',
  'dataset': ['dssmdata', 'dssmdata'],
  'testset': ['dssmdata', 'dssmdata'],
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
  'vedioattr_size': 2000  # 8368
}

param2 = {
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp_neg/',
  'modelpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp_neg/model2/',

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
  'keep_prob': 0.5,

  'emb_size': 128,
  'userattr_size': 706,
  'vedioattr_size': 8369
}

param3 = {
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp_neg/',
  'modelpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp_neg/model/',

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
  'keep_prob': 0.5,

  'emb_size': 128,
  'userattr_size': 706,
  'vedioattr_size': 8369
}


#param.update(param2)
#param.update(param3)

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
  embed_userattr = VedioMatchUtils.add_embedding(param['userattr_size'], param['emb_size'], '', 1,
                                                 emb_name="embed_userattr")
  embed_vedioattr = VedioMatchUtils.add_embedding_with_zero(param['vedioattr_size'], param['emb_size'], '', 1,
                                                            emb_name="embed_vedioattr")

  ##----------------------------input
  with tf.name_scope('input') as scope:
    input_user_base = tf.placeholder(shape=[None, 4], dtype='int32', name='input_user_base')
    input_user_cctype = tf.sparse_placeholder(tf.int32, name='input_user_cctype')
    input_user_cclassid1 = tf.sparse_placeholder(tf.int32, name='input_user_cclassid1')
    input_user_cclassid2 = tf.sparse_placeholder(tf.int32, name='input_user_cclassid2')
    input_user_ctag = tf.sparse_placeholder(tf.int32, name='input_user_ctag')

    input_item_vinfo = tf.placeholder(shape=[None, 3], dtype='int32', name='input_item_vinfo')
    input_item_ctag = tf.sparse_placeholder(tf.int32, name='input_item_ctag')

    label = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='input_label')

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

      embedding_item_vinfo = tf.gather(embed_vedioattr, input_item_vinfo, name='embedding_item_vinfo')
      embedding_item_vinfo = tf.reshape(embedding_item_vinfo, [-1, 3 * param['emb_size']])
      embedding_item_ctag = tf.nn.embedding_lookup_sparse(embed_vedioattr, input_item_ctag, None, combiner='mean',
                                                         name='embedding_item_ctag')

  ##----------------------------concat layer
  with tf.name_scope('concat') as scope:
    concat_item = tf.concat([embedding_item_vinfo, embedding_item_ctag], 1)
    concat_user = tf.concat(
      [embedding_user_base, embedding_user_cctype, embedding_user_cclassid1, embedding_user_cclassid2,
       embedding_user_ctag], 1)

  ##----------------------------fc layer
  with tf.name_scope('fc') as scope:
    fc_user_w0, fc_user_b0 = VedioMatchUtils.create_w_b(8 * param['emb_size'], param['emb_size'] * 2,
                                                        w_name="fc_user_w0", b_name="fc_user_b0")
    fc_user_w1, fc_user_b1 = VedioMatchUtils.create_w_b(param['emb_size'] * 2, param['emb_size'], w_name="fc_user_w1",
                                                        b_name="fc_user_b1")

    fc_item_w0, fc_item_b0 = VedioMatchUtils.create_w_b(4 * param['emb_size'], param['emb_size'] * 2,
                                                        w_name="fc_item_w0", b_name="fc_item_b0")
    fc_item_w1, fc_item_b1 = VedioMatchUtils.create_w_b(param['emb_size'] * 2, param['emb_size'], w_name="fc_item_w1",
                                                        b_name="fc_item_b1")

    user_z0 = VedioMatchUtils.calculate_y(concat_user, fc_user_w0, fc_user_b0, keep_prob)
    user_z1 = VedioMatchUtils.calculate_y(user_z0, fc_user_w1, fc_user_b1, keep_prob)

    item_z0 = VedioMatchUtils.calculate_y(concat_item, fc_item_w0, fc_item_b0, keep_prob)
    item_z1 = VedioMatchUtils.calculate_y(item_z0, fc_item_w1, fc_item_b1, keep_prob)

    user_z1_n = tf.nn.l2_normalize(user_z1, 1)
    item_z1_n = tf.nn.l2_normalize(item_z1, 1)

    cos_similarity=tf.reduce_sum(tf.multiply(user_z1_n, item_z1_n), 1, keep_dims=True)
    pred=tf.sigmoid(cos_similarity)

  ##----------------------------loss layer
  with tf.name_scope('loss') as scope:
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=cos_similarity))
    #loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=label, y_pred=pred))
    loss = tf.contrib.losses.log_loss(pred, label, weights=1.0, epsilon=1e-07, scope=None)
    #loss = tf.reduce_mean(cos_similarity)
    learning_rate = tf.train.exponential_decay(0.005, global_step, param['decay_steps'], 0.98)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
  # grads_and_vars = optimizer.compute_gradients(loss)
  #    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  with tf.name_scope('summary') as scope:
    # Summary.
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summaries.append(tf.summary.scalar("learning_rate", learning_rate))
    summary_op = tf.summary.merge(summaries)

  readdata = VedioDSSMInputAnsy(param)
  debugdata = readdata.read_testdata_batch(size=5)

  debug_dict = {
    label: debugdata['label'],

    input_user_base: debugdata['user_base'],
    input_user_cctype: debugdata['user_cctype'],
    input_user_cclassid1: debugdata['user_cclassid1'],
    input_user_cclassid2: debugdata['user_cclassid2'],
    input_user_ctag: debugdata['user_ctag'],

    input_item_vinfo: debugdata['item_vinfo'],
    input_item_ctag: debugdata['item_ctag'],

    keep_prob: 1.0,
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
        label: train_data['label'],

        input_user_base: train_data['user_base'],
        input_user_cctype: train_data['user_cctype'],
        input_user_cclassid1: train_data['user_cclassid1'],
        input_user_cclassid2: train_data['user_cclassid2'],
        input_user_ctag: train_data['user_ctag'],

        input_item_vinfo: train_data['item_vinfo'],
        input_item_ctag: train_data['item_ctag'],

        keep_prob: param['keep_prob'],
        global_step: step
      }

      if (step > 0 and step % param['test_batch'] == 0):
        ## run optimizer
        _, l, s, lr = session.run([optimizer, loss, summary_op, learning_rate], feed_dict=feed_dict)
        print('[%s]\t[Train]\tIter:%d\tloss=%.6f\tlr=%.6f' % (
        time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), step, l, lr), end='\n')

        summary_writer.add_summary(s, step)

        lv, user_z1_out, item_z1_out, item_i, user_i, cos_similarity_out, pred_out = \
          session.run([loss, user_z1_n, item_z1_n, concat_item, concat_user, cos_similarity, pred], \
                      feed_dict=debug_dict)
        VedioMatchUtils.print_model_data(lv, user_z1_out, item_z1_out, item_i, user_i, cos_similarity_out, pred_out)

        test_data = readdata.read_testdata_batch_ansyc()
        feed_dict = {
          label: test_data['label'],

          input_user_base: test_data['user_base'],
          input_user_cctype: test_data['user_cctype'],
          input_user_cclassid1: test_data['user_cclassid1'],
          input_user_cclassid2: test_data['user_cclassid2'],
          input_user_ctag: test_data['user_ctag'],

          input_item_vinfo: test_data['item_vinfo'],
          input_item_ctag: test_data['item_ctag'],

          keep_prob: 1.0,
          global_step: step
        }

        ## get acc
        lv, user_z1_out, item_z1_out, cos_similarity_out, pred_out = \
          session.run([loss, user_z1_n, item_z1_n, cos_similarity, pred], feed_dict=feed_dict)

        acc, prec, rec, auc = VedioMatchUtils.get_accprecrecauc(pred_out, test_data['label'], cos_similarity_out)
        print('[Test]\tIter:%d\tloss=%.6f\taccuracy=%.6f\tprecision=%.6f\trecall=%.6f\tauc=%.6f' % (step, lv, acc, prec, rec, auc), end='\n')
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
