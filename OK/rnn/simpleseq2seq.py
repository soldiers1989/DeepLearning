#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code from https://github.com/zhuanxuhit/nd101/blob/master/1.Intro_to_Deep_Learning/11.How_to_Make_a_Language_Translator/1-seq2seq.ipynb
import tensorflow as tf
import numpy as np

def getbatch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            getbatch-sized list of integers specifying amount of active
            time steps in each input sequence
    """
    
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths

def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
        raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    
    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]
        
x = [[5, 7, 8], [6, 3], [3], [1]]
xt, xlen = getbatch(x)

xt
#Out[8]: 
#array([[5, 6, 3, 1],
#       [7, 3, 0, 0],
#       [8, 0, 0, 0]])

xlen
#Out[9]: [3, 2, 1, 1]

PAD = 0
EOS = 1
# UNK = 2
# GO  = 3

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

embeddings = tf.Variable(tf.truncated_normal([vocab_size, input_embedding_size], mean=0.0, stddev=0.1), dtype=tf.float32)

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
#我们系统的输入encoder_inputs和decoder_inputs都是 [decoder_max_time, batch_size]的形状
#但是我们encoder和decoder的输入形状都是要[max_time, batch_size, input_embedding_size]，
#因此我们需要对我们的是输入做一个word embedded

print(encoder_inputs)
# Tensor("encoder_inputs_1:0", shape=(?, ?), dtype=int32)

print(encoder_inputs_embedded)
# Tensor("embedding_lookup:0", shape=(?, ?, 20), dtype=float32)

encoder_cell = tf.contrib.rnn.BasicLSTMCell(encoder_hidden_units)
lstm_layers = 2
#cell = tf.contrib.rnn.MultiRNNCell([encoder_cell] * lstm_layers)
cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(encoder_hidden_units) for _ in range(lstm_layers)])

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell,encoder_inputs_embedded,dtype=tf.float32,time_major=True)

print(encoder_outputs)
#Tensor("rnn/TensorArrayStack/TensorArrayGatherV3:0", shape=(?, ?, 20), dtype=float32)

print(encoder_final_state)
#(LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_2:0' shape=(?, 20) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_3:0' shape=(?, 20) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_4:0' shape=(?, 20) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_5:0' shape=(?, 20) dtype=float32>))

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    batch_ = [[6], [3, 4], [9, 8, 7]]
#
#    batch_, batch_length_ = getbatch(batch_)
#    print('batch_:\n' + str(batch_))
#    # [[6 3 9] [0 4 8] [0 0 7]]
#    print('batch_length_:\n' + str(batch_length_))
#    # [1, 2, 3]
#
#    output_ = sess.run(encoder_outputs, feed_dict={ encoder_inputs: batch_ })
#    print('encode output:\n' + str(output_))
#    # [[[ 1.07450760e-03 -1.20031589e-03 -1.00758066e-03  1.47092890e-03 -2.26932392e-03 -2.55039101e-03 -1.40686368e-03  1.57312228e-04...
#    print('encode output shape:\n' + str(output_.shape))  #(3, 3, 20)

decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets') #Tensor("decoder_targets:0", shape=(?, ?), dtype=int32)
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs) #Tensor("embedding_lookup_1:0", shape=(?, ?, 20), dtype=float32)

#decoder_cell = tf.contrib.rnn.BasicLSTMCell(decoder_hidden_units)
decoder = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(encoder_hidden_units) for _ in range(lstm_layers)])

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder, decoder_inputs_embedded,
    initial_state=encoder_final_state,
    dtype=tf.float32, time_major=True, scope="plain_decoder",
)

print(decoder_outputs)  #Tensor("plain_decoder/TensorArrayStack/TensorArrayGatherV3:0", shape=(?, ?, 20), dtype=float32) input_embedding_size=20
print(decoder_final_state) 
#(LSTMStateTuple(c=<tf.Tensor 'plain_decoder/while/Exit_2:0' shape=(?, 20) dtype=float32>, h=<tf.Tensor 'plain_decoder/while/Exit_3:0' shape=(?, 20) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'plain_decoder/while/Exit_4:0' shape=(?, 20) dtype=float32>, h=<tf.Tensor 'plain_decoder/while/Exit_5:0' shape=(?, 20) dtype=float32>))

decoder_logits = tf.contrib.layers.fully_connected(decoder_outputs,  # shape=(?, ?, 20)
                                                   vocab_size,       # vocab_size = 10
                                                   activation_fn=None,
                                                   weights_initializer = tf.truncated_normal_initializer(stddev=0.1),
                                                   biases_initializer=tf.zeros_initializer())
#对于RNN的输出，其shape是：[max_time, batch_size, hidden_units]，通过一个FC，将其映射为：[max_time, batch_size, vocab_size]
print(decoder_logits) #Tensor("fully_connected/BiasAdd:0", shape=(?, ?, 10), dtype=float32)

decoder_prediction = tf.argmax(decoder_logits, 2) # 在这一步我突然意识到了axis的含义。。。表明的竟然是在哪个维度上求 argmax。
print(decoder_prediction) #Tensor("ArgMax:0", shape=(?, ?), dtype=int64)

learn_rate = tf.placeholder(tf.float32)
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)
loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_ = [[6], [3, 4], [9, 8, 7]]

    batch_, batch_length_ = getbatch(batch_)
    print('batch_encoded:\n' + str(batch_))
    #[[6 3 9] [0 4 8]  [0 0 7]]
 
    din_, dlen_ = getbatch(np.ones(shape=(3, 1), dtype=np.int32), max_sequence_length=4)
    print('decoder inputs:\n' + str(din_))
    #[[1 1 1] [0 0 0] [0 0 0] [0 0 0]]

    pred_ = sess.run(decoder_prediction,
        feed_dict={
            encoder_inputs: batch_,
            decoder_inputs: din_,
            learn_rate:0.1
        })
    print('decoder predictions:\n' + str(pred_))
    #[[0 9 2] [0 3 2] [0 3 2] [0 2 2]]
    
print("build graph ok!")

batch_size = 100

batches = random_sequences(length_from=3, length_to=8, vocab_lower=2, vocab_upper=10, batch_size=batch_size)

print('head of the getbatch:')
for seq in next(batches)[:10]: print(seq)
  
# 当encoder_inputs 是[5, 6, 7]是decoder_targets是 [5, 6, 7, 1],1代表的是EOF，decoder_inputs则是 [1, 5, 6, 7]
def next_feed():
    thisbatch = next(batches)
    encoder_inputs_, _ = getbatch(thisbatch)
    decoder_targets_, _ = getbatch(
        [(sequence) + [EOS] for sequence in thisbatch]
    )
    decoder_inputs_, _ = getbatch(
        [[EOS] + (sequence) for sequence in thisbatch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }

loss_track = []
max_batches = 3001
batches_in_epoch = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        for batch in range(max_batches):
            fd = next_feed()
            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)

            if batch == 0 or batch % batches_in_epoch == 0:
                print('getbatch {}'.format(batch))
                print('  minibatch loss: {}'.format(sess.run(loss, fd)))
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                    print('  sample {}:'.format(i + 1))
                    print('    input     > {}'.format(inp))
                    print('    predicted > {}'.format(pred))
                    if i >= 2:
                        break
                print()
    except KeyboardInterrupt:
        print('training interrupted')