import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib.tensorboard.plugins import projector

logdir='log'
# load model
df_train = pd.read_csv(
    tf.gfile.Open('.'+os.sep+'data'+os.sep+'msg.pred'),
    skipinitialspace=True, 
    engine="python",
    delimiter=' ',
    skiprows=0)

df_train.columns = ['key']+[str(x) for x in range(100)]

uinlist=df_train['key']
del df_train['key']
embedding=df_train.values

if not os.path.exists(logdir):
    os.makedirs(logdir)

# setup a TensorFlow session
tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=embedding.shape)
set_x = tf.assign(X, place, validate_shape=False) #
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: embedding})

# write labels
tsvfile=os.path.join(logdir, 'metadata.tsv')
with open(tsvfile, 'w', encoding='utf-8') as f:
    for uin in uinlist:
        f.write(str(uin) + '\n')

# create a TensorFlow summary writer
summary_writer = tf.summary.FileWriter(logdir, sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = os.path.join('metadata.tsv')
#embedding_conf.metadata_path = os.path.join('metadata.tsv')
projector.visualize_embeddings(summary_writer, config)

# save the model
saver = tf.train.Saver()
saver.save(sess, os.path.join(logdir, "model.ckpt"))
