#coding: utf-8

import numpy as np

np.random.seed(171)

root_path = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/bincai/hot_mp/src/'
stopword_dir = root_path + 'stop_word/'
vector_path = root_path + 'vector/word2vector.txt'
path = root_path + 'data2/'
modelpath = path + 'model/'
vocab_path = modelpath + 'vocab.txt'
model_type = 'dnn'
token_type = 'word' #char #word

train_path=path+'train_add.txt' #'train_char.clean' #'train.clean'
test_path=path+'test.txt' #'yinghua_msgid_test.txt' #'test_big.clean' #'test_char.clean' #'test.clean'
max_title_length = 30
max_document_length = 1000
with_title = 0  #0: no-title 1: concat-title 2: only-title 3: title emb
max_frequency = -1
min_frequency = 1 # -1

valid_per = 0.2
max_vocab_size = 300000 #150000 #10000 #6292 200000
embed_size = 100 #300 #128
vec_size = 128
epochs=15
batch_size=256
seperate_type=1
is_pretrain_embed=1
train_embed=1
