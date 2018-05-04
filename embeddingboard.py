import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib.tensorboard.plugins import projector

logdir='log'
# load model
df_train = pd.read_csv(
    tf.gfile.Open('.'+os.sep+'data'+os.sep+'att.csv'),
    skipinitialspace=True, 
    engine="python",
    skiprows=0)
uinlist=df_train['uin']
del df_train['uin']
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
with open(tsvfile, 'w') as f:
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
