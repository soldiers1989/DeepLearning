# -*- coding: UTF-8 -*-
'''
model desc:
sparse feature: linear-part, fm-part, fm-hidden
dense feature: linear-part, fm-hidden
total = linear-part + fm-part + dnn-part
reg: embedding reg(only reg on used features) + non-embedding reg(L2 norm)
@author: rgchen
'''


import time
import tensorflow as tf
import math
import numpy as np
import re
from sklearn.metrics import roc_auc_score
from utils.deepFM_util_V5 import batch_inputs

## for multi-GPU training
TOWER_NAME = 'tower'
def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var

def _get_loss_collection_name(istrain):
    return "train_loss" if istrain else "test_loss"

def _variable_with_weight_decay(name, shape, stddev, wd = None, istrain = True):
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        _add_loss('weight_loss_%s' % name, var, wd, istrain)
    return var

def _add_loss(name, var, wd, istrain):
    l2loss = tf.multiply(tf.nn.l2_loss(var), wd, name=name)
    loss_name = _get_loss_collection_name(istrain)
    tf.add_to_collection(loss_name, l2loss)

def inference(features, params, istrain=True):
    loss_name = _get_loss_collection_name(istrain)
    print("debug-info: len(tf.get_collection('%s')):" % loss_name, len(tf.get_collection(loss_name)))

    init_value = params['init_value']
    dim = params['dim']
    layer_sizes = params['layer_sizes']
    field_sizes = params['field_sizes']
    field_types = params['field_types']
    field_iskeeps = params['field_iskeeps']

    sparse_ids = [i for (i, field_type) in enumerate(field_types) if field_type == "sparse" and field_iskeeps[i]]
    dense_ids = [i for (i, field_type) in enumerate(field_types) if field_type == "dense" and field_iskeeps[i]]
    dense_feature_cnt = sum([field_sizes[i] for i in dense_ids])
    sparse_feature_cnt = sum([field_sizes[i] for i in sparse_ids])
    feature_cnt = sparse_feature_cnt + dense_feature_cnt
    print("debug-info: sparse_ids:%s" % sparse_ids)
    print("debug-info: dense_ids:%s" % dense_ids)
    print("debug-info: dense_feature_cnt:%s" % dense_feature_cnt)
    print("debug-info: sparse_feature_cnt:%s" % sparse_feature_cnt)
    print("debug-info: feature_cnt:%s" % feature_cnt)

    # ------------------------------
    # part 1: linear part
    # ------------------------------
    bias = _variable_with_weight_decay('bias', [1], init_value, None, istrain)
    preds = bias
    parts = [bias]
    if params['is_use_linear_dense']:
        for i in dense_ids:
            w_dense_i = _variable_with_weight_decay('w_linear_dense_%s' % i,
                                                     [field_sizes[i], 1],
                                                     init_value,
                                                     params['reg_w_linear'],
                                                     istrain)
            linear_dense_part = tf.matmul(features['dense_values_%s' % i], w_dense_i, name = 'linear_dense_%s' % i)
            parts.append(linear_dense_part)
            preds += linear_dense_part
    if params['is_use_linear_sparse']:
        for i in sparse_ids:
            w_sparse_i = _variable_with_weight_decay('w_linear_sparse_%s' % i,
                                                     [field_sizes[i], 1],
                                                     init_value,
                                                     None,
                                                     istrain)
            linear_sparse_part = tf.nn.embedding_lookup_sparse(w_sparse_i,
                                                   features['sparse_indices_%s' % i],
                                                   features['sparse_values_%s' % i],
                                                   combiner="sum",
                                                   name='linear_sparse_%s' % i)
            preds += linear_sparse_part
            parts.append(linear_sparse_part)
            # _add_loss('embed_loss_linear_sparse_%s' % i, linear_sparse_part, params['reg_w_linear'], istrain) # (a*a+b*b)^2
            _add_loss('embed_loss_linear_sparse_%s' % i, w_sparse_i, params['reg_w_linear']/np.sqrt(field_sizes[i]), istrain) # embed^2
            # _add_loss('embed_loss_linear_sparse_%s' % i, tf.nn.embedding_lookup_sparse(w_sparse_i,features['sparse_indices_%s' % i]), params['reg_w_linear'], istrain) # a*a + b*b
    print("debug-info: linear part: preds.shape", preds.shape)

    # ------------------------------
    # part 2: FM part
    # ------------------------------
    w_fm_map, fm_hidden_map = {}, {}
    for i in sparse_ids:
        w_fm_map[i] = _variable_with_weight_decay('w_fm_sparse_%s' % i, [field_sizes[i], dim], init_value, None, istrain)
        fm_hidden_map[i] = tf.nn.embedding_lookup_sparse(w_fm_map[i],
                                                         features['sparse_indices_%s' % i],
                                                         features['sparse_values_%s' % i],
                                                         combiner = "sum")
        # _add_loss('embed_loss_fm_%s' % i, fm_hidden_map[i], params['reg_w_fm'], istrain)
        _add_loss('embed_loss_fm_%s' % i, w_fm_map[i], params['reg_w_fm']/np.sqrt(field_sizes[i]), istrain)
        # _add_loss('embed_loss_fm_%s' % i, tf.nn.embedding_lookup_sparse(w_fm_map[i],features['sparse_indices_%s' % i]), params['reg_w_fm'], istrain)

    if params['is_use_fm_part']:
        fm_hidden_sum = sum(fm_hidden_map.values())
        sparse_values_square_map = {i : tf.SparseTensor(features['sparse_values_%s' % i].indices,
                                                        tf.pow(features['sparse_values_%s' % i].values, 2),
                                                        features['sparse_values_%s' % i].dense_shape)
                                    for i in sparse_ids}
        xv2 = tf.concat([tf.nn.embedding_lookup_sparse(tf.pow(w_fm_map[i],2),
                                                       features['sparse_indices_%s' % i],
                                                       sparse_values_square_map[i],
                                                       combiner="sum")
                         for i in sparse_ids], 1)
        fm_part = 0.5 * tf.reduce_sum(tf.pow(fm_hidden_sum, 2), 1, keep_dims=True) - 0.5 * tf.reduce_sum(xv2, 1, keep_dims=True)
        print("debug-info: fm part: fm_part.shape", fm_part.shape)

    # ------------------------------
    # part 3: DNN part
    # ------------------------------
    if params['is_use_dnn_part']:
        dense_values = tf.concat( [features['dense_values_%s' % i] for i in dense_ids], 1)
        print('dense_values.shape',dense_values.shape)

        if params['is_use_dense_in_fm_layer']:
            fm_input = tf.concat([fm_hidden_map[i] for i in sparse_ids] + [dense_values], 1)
            last_layer_size = len(sparse_ids) * dim + dense_feature_cnt
        else:
            fm_input = tf.concat([fm_hidden_map[i] for i in sparse_ids], 1)
            last_layer_size = len(sparse_ids) * dim

        hidden_nn_layers = []
        hidden_nn_layers.append(fm_input)

        layer_idx = 0
        for layer_size in layer_sizes:
            cur_w_nn_layer = _variable_with_weight_decay('w_nn_layer_' + str(layer_idx),
                                                         [last_layer_size, layer_size],
                                                         1/np.sqrt(last_layer_size/2.),
                                                         params['reg_w_nn'], istrain)
            cur_b_nn_layer = _variable_with_weight_decay('b_nn_layer_' + str(layer_idx),
                                                         [layer_size],
                                                         init_value, None,  istrain)
            cur_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx], cur_w_nn_layer, cur_b_nn_layer)

            if params['activations'][layer_idx]=='tanh':
                cur_hidden_nn_layer = tf.nn.tanh(cur_hidden_nn_layer)
            elif params['activations'][layer_idx]=='sigmoid':
                cur_hidden_nn_layer = tf.nn.sigmoid(cur_hidden_nn_layer)
            elif params['activations'][layer_idx]=='relu':
                cur_hidden_nn_layer = tf.nn.relu(cur_hidden_nn_layer)

            hidden_nn_layers.append(cur_hidden_nn_layer)

            layer_idx += 1
            last_layer_size = layer_size

        if params['is_use_dense_in_last_layer']:
            hidden_nn_layers[-1] = tf.concat([hidden_nn_layers[-1], dense_values], 1)
            last_layer_size += dense_feature_cnt
        w_nn_output = _variable_with_weight_decay('w_nn_output',
                                                  [last_layer_size, 1],
                                                  1/np.sqrt(last_layer_size/2.),
                                                  params['reg_w_nn'], istrain)
        dnn_part = tf.matmul(hidden_nn_layers[-1], w_nn_output, name = 'dnn_part')
        print("debug-info: dnn part: dnn_part.shape", dnn_part.shape)


    if params['is_use_fm_part'] and params['is_use_dnn_part']:
        preds += fm_part + dnn_part
        parts += [fm_part, dnn_part]
    elif params['is_use_fm_part']:
        preds += fm_part
        parts += [fm_part]
    elif params['is_use_dnn_part']:
        preds += dnn_part
        parts += [dnn_part]

    part_prefix = "train-" if istrain else "test-"
    for var in parts:
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', var.op.name)
        tf.summary.scalar(part_prefix+"parts/"+loss_name, tf.reduce_mean(var))
    tf.summary.scalar(part_prefix+"parts/preds", tf.reduce_mean(preds))
    checks = [bias]
    return preds, checks # checks is only for debug

def loss(logits, labels, params, istrain):
    if params['loss'] == 'cross_entropy_loss':
        labels = tf.cast(labels, tf.float32)
        logits = tf.reshape(logits,[-1])
        labels = tf.reshape(labels,[-1])
        error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    elif params['loss'] == 'square_loss':
        preds = tf.sigmoid(logits)
        labels = tf.cast(labels, tf.float32)
        error = tf.reduce_mean(tf.squared_difference(preds, labels))
    else:
        raise ValueError("%s not define!" % params['loss'])

    loss_name = _get_loss_collection_name(istrain)
    tf.add_to_collection(loss_name, error)
    total_loss = tf.add_n(tf.get_collection(loss_name), name='total_%s' % loss_name)
    return total_loss


def tower_loss(scope, features, params, istrain = True):
    loss_name = _get_loss_collection_name(istrain)
    logits, checks = inference(features, params, istrain)
    _ = loss(logits, features['label'], params, istrain)
    losses = tf.get_collection(loss_name, scope)

    total_loss = tf.add_n(losses, name=loss_name)
    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        tf.summary.scalar("loss-"+loss_name, l)
        print("istrain: %s, l.op.name: %s, loss_name: %s" % (str(istrain), l.op.name, loss_name))
    return logits, total_loss, checks

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def single_run(params):
    print (params)
    print('start single_run')

    field_sizes, field_types = params['field_sizes'], params['field_types']
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        tf.set_random_seed(1)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        lr = tf.train.exponential_decay(params['learning_rate'],
                                        global_step,
                                        5000,
                                        0.95,
                                        staircase=True)
        if params['optimizer']=='adadelta':
            opt = tf.train.AdadeltaOptimizer(lr)
        elif params['optimizer']=='sgd':
            opt = tf.train.GradientDescentOptimizer(lr)
        elif params['optimizer']=='adam':
            opt = tf.train.AdamOptimizer(lr)
        elif params['optimizer']=='ftrl':
            opt = tf.train.FtrlOptimizer(lr)
        else:
            opt = tf.train.GradientDescentOptimizer(params['learning_rate'], global_step)

        tower_grads = []
        trainfeatures = batch_inputs(params['train_file'], field_sizes, field_types, batch_size=params['batch_size'], num_preprocess_threads=1)
        testfeatures  = batch_inputs(params['test_file'],  field_sizes, field_types, batch_size=params['batch_size'], num_preprocess_threads=1)
        if params['output_predictions']:
            testfeatures  = batch_inputs(params['test_file'],  field_sizes, field_types, num_epochs = 1, batch_size=params['batch_size'], num_preprocess_threads=1)
        testlabellist, testlogitslist, testlosslist = [], [], []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(params['GPU_NUM']):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                        print('one-step')
                        # for train
                        logits, loss, checks = tower_loss(scope, trainfeatures, params)
                        tf.get_variable_scope().reuse_variables()

                        # for test
                        testlogits, testloss, testchecks = tower_loss(scope, testfeatures, params, istrain = False)
                        testlogitslist.append(testlogits)
                        testlabellist.append(testfeatures['label'])

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)
                        # for predict

        # for train
        grads = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        train_op = apply_gradient_op

        summaries.append(tf.summary.scalar("loss/batch_train_loss", loss))
        summaries.append(tf.summary.scalar("loss/batch_test_loss" , testloss))
        summaries.append(tf.summary.scalar('learning_rate', lr))
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
            summaries.append(tf.summary.scalar("all_l2_loss/" + var.op.name, tf.nn.l2_loss(var)))
        # for var in checks:
        #     summaries.append(tf.summary.scalar("all_attr/" + var.op.name, tf.nn.l2_loss(var)))

        summary_op = tf.summary.merge(summaries)

        #saver = tf.train.Saver()
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))

        tf.train.start_queue_runners(sess=sess)
        print('-'*20)
        sess.run(init)
        print('*'*20)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(params['log_path'], graph=sess.graph)

        ma_loss, ma_test_loss, ma_test_auc, duration = 0,0,0,0
        for step in range(150000):
            duration -= time.time()
            # feats_tr, logits_tr, checks_tr = sess.run([trainfeatures, logits, checks])
            _, loss_value = sess.run([train_op, loss])
            ma_loss = ma_loss*0.95 + loss_value*0.05 if ma_loss>0 else loss_value
            duration += time.time()

            if step % 300 == 0:
                testlabels_value, testlogits_value, testlosses_value = [], [], []
                while True:
                    labels_te, logits_te, loss_te = sess.run([testlabellist, testlogitslist, testloss])
                    testlabels_value.extend(np.vstack(labels_te).reshape(-1).tolist())
                    testlogits_value.extend(np.vstack(logits_te).reshape(-1).tolist())
                    testlosses_value.append(float(loss_te))
                    # 超过3k个样本再算auc
                    if len(testlabels_value) > 6000:
                        batch_auc = roc_auc_score(testlabels_value, testlogits_value)
                        batch_loss = sum(testlosses_value) / len(testlosses_value)
                        ma_test_auc = ma_test_auc*0.9 + batch_auc*0.1 if ma_test_auc > 0 else batch_auc
                        ma_test_loss = ma_test_loss*0.9 + batch_loss*0.1 if ma_test_loss > 0 else batch_loss
                        break
                print ('len(testlabels_value)', len(testlabels_value))
                print('train steps: %s, totaltime: %.3f min' % (step, duration/60.0))
                print('train_loss: %3f, test_loss: %.3f, test_auc: %.3f' % (ma_loss, ma_test_loss, ma_test_auc))

                # summary info
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="ma_train_loss", simple_value=float(ma_loss)),
                    tf.Summary.Value(tag="ma_test_loss", simple_value=float(ma_test_loss)),
                    tf.Summary.Value(tag="ma_test_auc", simple_value=ma_test_auc),
                ])
                summary_writer.add_summary(summary, step)
                summary_writer.add_summary(sess.run(summary_op), step)
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        summary_writer.close()
        coord.request_stop()
        coord.join(threads)

