#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import VedioMatchUtils
from VedioMatchInputAnsy import VedioMatchInputAnsy
import os
import sys
import math
import tensorflow as tf
import numpy as np
import time

param = {
  'inputpath': 'data/',
  'dataset': ['part-00000', 'part-00000'],
  'testset': ['part-00000', 'part-00000'],
  'predset': [],

  'shuffle_file': True,
  'batch_size': 10,
  'batch_size_test':256,
  'test_batch':100,
  'save_batch':500,
  'total_batch':1000,
  'decay_steps':1000,
  'keep_prob':0.8,

  'emb_size': 128,
  'userattr_size': 706,
  'vedioattr_size': 2000, # 8368
  'margin': 0.25,
}

param2 = {
  'inputpath': '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp_mv/',
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
  'batch_size_test':10240,
  'test_batch': 500,
  'save_batch': 5000,
  'total_batch': 100000,
  'decay_steps': 1000,
  'keep_prob': 0.8,

  'emb_size': 128,
  'userattr_size': 706,
  'vedioattr_size': 8368,
  'margin': 0.25,
}

param.update(param2)

def main():
#  ## define sparse tensor shape
#  sparse_shape    = np.array([batch_size, input_max_size], dtype=np.int32)

#  wxbiz_offline_db::daily_mid_wxsearch_kyk_video_log_share212_join_vedio_user_portrait_timeline_train_mv
#  ds,uin,gender,age,grade,region_code,clickctype,clickclass_id_1,clickclass_id_2,clicktag,noclickctype,noclickclass_id_1,noclickclass_id_2,noclicktag
#  t1vid,t1title,t1title0,t1class_id_1,t1class_id_2,t1tag
#  t2vid,t2title,t2title1,t2class_id_1,t2class_id_2,t2tag
  #用户
  #用tf.nn.embedding_lookup gender, age, grade, region_code
  #用tf.nn.embedding_lookup_sparse  clickctype
  #用tf.nn.embedding_lookup_sparse  clickclass_id_1
  #用tf.nn.embedding_lookup_sparse  clickclass_id_2
  #用tf.nn.embedding_lookup_sparse  clicktag
  #用tf.nn.embedding_lookup_sparse  noclickctype
  #用tf.nn.embedding_lookup_sparse  noclickclass_id_1
  #用tf.nn.embedding_lookup_sparse  noclickclass_id_2
  #用tf.nn.embedding_lookup_sparse  noclicktag
  #
  #视频
  #用tf.nn.embedding_lookup t1title0,t1class_id_1,t1class_id_2
  #用tf.nn.embedding_lookup_sparse t1tag

  ## define embeddings
  # 个人信息没有0
  embed_userattr = VedioMatchUtils.add_embedding(param['userattr_size'], param['emb_size'], '', 1, emb_name="embed_userattr")
  embed_vedioattr = VedioMatchUtils.add_embedding_with_zero(param['vedioattr_size'], param['emb_size'], '', 1, emb_name="embed_userattr")

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
      embedding_user_base=tf.reshape(embedding_user_base, [-1, 4*param['emb_size']])
      embedding_user_cctype = tf.nn.embedding_lookup_sparse(embed_vedioattr, input_user_cctype, None, combiner='mean', name='embedding_user_cctype')
      embedding_user_cclassid1 = tf.nn.embedding_lookup_sparse(embed_vedioattr, input_user_cclassid1, None, combiner='mean', name='embedding_user_cclassid1')
      embedding_user_cclassid2 = tf.nn.embedding_lookup_sparse(embed_vedioattr, input_user_cclassid2, None, combiner='mean', name='embedding_user_cclassid2')
      embedding_user_ctag = tf.nn.embedding_lookup_sparse(embed_vedioattr, input_user_ctag, None, combiner='mean', name='embedding_user_ctag')

      embedding_pos_vinfo = tf.gather(embed_vedioattr, input_pos_vinfo, name='embedding_pos_vinfo')
      embedding_pos_vinfo=tf.reshape(embedding_pos_vinfo, [-1, 3*param['emb_size']])
      embedding_pos_ctag = tf.nn.embedding_lookup_sparse(embed_vedioattr, input_pos_ctag, None, combiner='mean', name='embedding_pos_ctag')

      embedding_neg_vinfo = tf.gather(embed_vedioattr, input_neg_vinfo, name='embedding_neg_vinfo')
      embedding_neg_vinfo=tf.reshape(embedding_neg_vinfo, [-1, 3*param['emb_size']])
      embedding_neg_ctag = tf.nn.embedding_lookup_sparse(embed_vedioattr, input_neg_ctag, None, combiner='mean', name='embedding_neg_ctag')

  ##----------------------------concat layer
  with tf.name_scope('concat') as scope:
    concat_pos  = tf.concat([embedding_pos_vinfo, embedding_pos_ctag], 1)
    concat_neg  = tf.concat([embedding_neg_vinfo, embedding_neg_ctag], 1)
    concat_user = tf.concat([embedding_user_base, embedding_user_cctype, embedding_user_cclassid1, embedding_user_cclassid2, embedding_user_ctag], 1)

  ##----------------------------fc layer
  with tf.name_scope('fc') as scope:
    fc_user_w0, fc_user_b0 = VedioMatchUtils.create_w_b(8*param['emb_size'], param['emb_size'], w_name="fc_user_w0", b_name="fc_user_b0")
    fc_user_w1, fc_user_b1 = VedioMatchUtils.create_w_b(param['emb_size'], param['emb_size'], w_name="fc_user_w1", b_name="fc_user_b1")

    fc_item_w0, fc_item_b0 = VedioMatchUtils.create_w_b(4*param['emb_size'], param['emb_size'], w_name="fc_item_w0", b_name="fc_item_b0")
    fc_item_w1, fc_item_b1 = VedioMatchUtils.create_w_b(param['emb_size'], param['emb_size'], w_name="fc_item_w1", b_name="fc_item_b1")

    user_z0 = VedioMatchUtils.calculate_y(concat_user, fc_user_w0, fc_user_b0, keep_prob)
    user_z1 = VedioMatchUtils.calculate_y(user_z0, fc_user_w1, fc_user_b1, keep_prob)

    pos_z0 = VedioMatchUtils.calculate_y(concat_pos, fc_item_w0, fc_item_b0, keep_prob)
    pos_z1 = VedioMatchUtils.calculate_y(pos_z0, fc_item_w1, fc_item_b1, keep_prob)

    neg_z0 = VedioMatchUtils.calculate_y(concat_neg, fc_item_w0, fc_item_b0, keep_prob)
    neg_z1 = VedioMatchUtils.calculate_y(neg_z0, fc_item_w1, fc_item_b1, keep_prob)

    user_z1_n = tf.nn.l2_normalize(user_z1, 1)
    pos_z1_n  = tf.nn.l2_normalize(pos_z1, 1)
    neg_z1_n  = tf.nn.l2_normalize(neg_z1, 1)

  ##----------------------------loss layer
  with tf.name_scope('loss') as scope:
    loss = VedioMatchUtils.compute_triplet_loss(user_z1_n, pos_z1_n, neg_z1_n, param['margin'])
    learning_rate = tf.train.exponential_decay(0.0005, global_step, param['decay_steps'], 0.98)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#    learning_rate_emb   = tf.train.exponential_decay(0.0005, global_step, param['decay_steps'], 0.98)
#    var_emb = [embed_userattr, embed_vedioattr]
#    optimizer_emb = tf.train.GradientDescentOptimizer(learning_rate_emb).minimize(loss, var_list = var_emb)
#
#    learning_rate_fc = tf.train.exponential_decay(0.001, global_step, param['decay_steps'], 0.98)
#    var_fc = [fc_user_w0, fc_user_b0, fc_user_w1, fc_user_b1, fc_item_w0, fc_item_b0, fc_item_w1, fc_item_b1]
#    optimizer_fc  = tf.train.GradientDescentOptimizer(learning_rate_fc).minimize(loss, var_list = var_fc)
#
#    optimizer = tf.group(optimizer_emb,  optimizer_fc)

  readdata = VedioMatchInputAnsy(param)
  readdata.start_ansyc()

  with tf.Session() as session:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
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

      ## run optimizer
      _, l= session.run([optimizer, loss], feed_dict=feed_dict)

      if (step>0 and step % param['test_batch'] == 0):
        print('[%s]\t[Train]\tIter:%d\tloss=%.6f' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), step, l), end='\n')
        
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
  
          keep_prob: param['keep_prob'],
          global_step: step
        }
        
        ## get acc
        lv, user_z1_out, pos_z1_out, neg_z1_out = \
            session.run([loss, user_z1_n, pos_z1_n, neg_z1_n], feed_dict=feed_dict )

        acc = VedioMatchUtils.get_acc(user_z1_out, pos_z1_out, neg_z1_out)
        print('[Test]\tIter:%d\tloss=%.6f\taccuracy=%.6f' % (step, lv, acc), end='\n')
        print("-----------------------------------------")
        
      if (step>0 and step % param['save_batch'] == 0):  
        model_name = "dnn-model-" + str(step)
        if not os.path.exists(param['inputpath']):
          os.mkdir(param['inputpath'])
        saver.save(session, os.path.join(param['inputpath'], model_name))

  readdata.stop_and_wait_ansyc()



#
#  batch =  tf.Variable(0)
#  learning_rate_emb   = tf.train.exponential_decay(0.0005, batch * batch_size, train_num, 0.9)
#  learning_rate_cnn   = tf.train.exponential_decay(0.005, batch * batch_size, train_num, 0.9)
#  learning_rate_fc    = tf.train.exponential_decay(0.002, batch * batch_size, train_num, 0.9)
#  ##
#  var_fc = [fc_w1, fc_b1, fc_w0, fc_b0]
#  optimizer_emb = tf.train.GradientDescentOptimizer(learning_rate_emb).minimize(loss, var_list = [embeddings])
#  optimizer_fc  = tf.train.GradientDescentOptimizer(learning_rate_fc).minimize(loss, var_list = var_fc)
#  if title_method == 'cnn':
#    var_cnn = [cnn_w0, cnn_w1, cnn_w2, cnn_b0, cnn_b1, cnn_b2]
#    optimizer_cnn = tf.train.GradientDescentOptimizer(learning_rate_cnn).minimize(loss, var_list = var_cnn)
#    optimizer = tf.group(optimizer_emb, optimizer_cnn, optimizer_fc)
#  else:
#    optimizer = tf.group(optimizer_emb,  optimizer_fc)
#
#  with tf.Session() as session:
#    tf.global_variables_initializer().run()
#    saver = tf.train.Saver()
#    step = 0
#    for title_ids, tag_ids    , tag_val,\
#      pos_ids  , tag_pos_ids, tag_pos_val, \
#      neg_ids  , tag_neg_ids, tag_neg_val in \
#      data_generator.get_triplet_generator(data_dir=train_dir, epochs=train_epochs, tag_same=tag_same, is_train=True, max_iters=max_iters):
#
#      ## feed data to tf session
#      feed_dict = {
#        input_title_ids:title_ids,
#        input_pos_ids  :pos_ids,
#        input_neg_ids  :neg_ids,
#        input_tag_ids:    (tag_ids, tag_val, sparse_shape),
#        input_pos_tag_ids:(tag_pos_ids, tag_pos_val, sparse_shape),
#        input_neg_tag_ids:(tag_neg_ids, tag_neg_val, sparse_shape),
#        batch: step, keep_prob: 0.8
#      }
#
#      ## run optimizer
#      _, l= session.run([optimizer, loss], feed_dict=feed_dict)
#      print('[%s]\t[Train]\tIter:%d\tloss=%.6f' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), step, l), end='\n')
#
#      ## test model
#      step += 1
#      if (step % test_every_iters == 0) :
#        ## get test data
#        title_ids,tag_ids,tag_val, \
#        pos_ids,tag_pos_ids,tag_pos_val, \
#        neg_ids,tag_neg_ids,tag_neg_val \
#          = data_shuffler.get_triplet(n_triplets=batch_size, tag_same=tag_same, is_target_set_train=False, max_line_size=sentence_len)
#
#        ## feed session
#        f_dict = {
#               input_title_ids:title_ids,
#               input_pos_ids:pos_ids,
#               input_neg_ids:ne],  j2018/5/10 9:20:166XZx cvbg_ids,
#               input_tag_ids:(tag_ids, tag_val, sparse_shape),
#               input_pos_tag_ids:(tag_pos_ids, tag_pos_val, sparse_shape),
#               input_neg_tag_ids:(tag_neg_ids, tag_neg_val, sparse_shape),
#               batch: step, keep_prob: 1.0
#        }
#
#        ## get acc
#        lv, news_z1_out, pos_z1_out, neg_z1_out = \
#            session.run([loss, title_z1_n, pos_z1_n, neg_z1_n], feed_dict=f_dict)
#
#        acc = vrutils.get_acc(news_z1_out, pos_z1_out, neg_z1_out)
#        print('[Test]\tIter:%d\tloss=%.6f\taccuracy=%.6f' % (step, lv, acc), end='\n')
#        print("-----------------------------------------")
#
#      ## save model
#      if (step % save_every_iters == 0):
#        print('save model')
#        model_name = "dnn-model-" + str(step)
#        if not os.path.exists(model_dir):
#          os.mkdir(model_dir)
#        saver.save(session, os.path.join(model_dir, model_name))

if __name__ == "__main__":
  main()


