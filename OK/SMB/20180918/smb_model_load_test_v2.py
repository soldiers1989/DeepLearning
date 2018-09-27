#coding: utf-8
"""
dum_model_to_tf(): 将模型保存的TensorFlow Serving格式
load_tf_model: 尝试python的方式加载
"""
import os, sys
import numpy as np
import tensorflow as tf
from absl import flags

layer_sz = 2
signature_key = 'org_smb_signa' #'smb_signature'
input_keys = ['k','x','xlen']
outputs_key = ['p']

print(input_keys)
print(outputs_key)

#export_base = FLAGS.path + model_name
#export_dir = os.path.join(tf.compat.as_bytes("./model/org_smb_model"), tf.compat.as_bytes("1537340740")) 
#export_dir = os.path.join(tf.compat.as_bytes("./tmp/org_smb_model"), tf.compat.as_bytes("1537330375"))
#export_dir = "model/org_smb_model/1537346608"
export_dir = "model/org_smb_model/1537349117"
#file_dir="./20180910/model/org_smb_model"
#sess = tf.Session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
print(export_dir)
#meta_graph_def = tf.saved_model.loader.load(sess, ['org_smb_model'], file_dir)
meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir) #tf.saved_model.tag_constants.SERVING

signature = meta_graph_def.signature_def
print( [signature[signature_key].inputs[val].name for val in input_keys])
print( [signature[signature_key].outputs[val].name for val in outputs_key])

inputs_tensor = {val:sess.graph.get_tensor_by_name(signature[signature_key].inputs[val].name) for val in input_keys}
outputs_tensor = {val:sess.graph.get_tensor_by_name(signature[signature_key].outputs[val].name) for val in outputs_key}
#x = {inputs_tensor['x']:np.array([i for i in range(2)]).reshape([1, 2]),inputs_tensor['k']:1.0, inputs_tensor['xlen']:[2]}
x = {inputs_tensor['x']:np.array( [[1,43]]),inputs_tensor['k']:1.0, inputs_tensor['xlen']:[2]}
print (x)

out_y = sess.run([outputs_tensor["p"]], feed_dict=x)

print(len(out_y[0]))
print(out_y)   
    
    
    
    
