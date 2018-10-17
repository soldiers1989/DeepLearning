#! -*- coding: utf-8 -*-
from keras import backend as K
from keras import utils
from keras.datasets import mnist
from keras.layers import *
from keras.models import Model

from OK.misc.Capsule.Capsule_Keras import *

#准备训练数据
batch_size = 128
num_classes = 10
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

#准备自定义的测试样本
#对测试集重新排序并拼接到原来测试集，就构成了新的测试集，每张图片有两个不同数字
idx = [x for x in range(len(x_test))]
np.random.shuffle(idx)
X_test = np.concatenate([x_test, x_test[idx]], 1)
Y_test = np.vstack([y_test.argmax(1), y_test[idx].argmax(1)]).T
X_test = X_test[Y_test[:,0] != Y_test[:,1]] #确保两个数字不一样
Y_test = Y_test[Y_test[:,0] != Y_test[:,1]]
Y_test.sort(axis=1) #排一下序，因为只比较集合，不比较顺序

##搭建普通CNN分类模型
input_image = Input(shape=(None,None,1))
cnn = Conv2D(64, (3, 3), activation='relu')(input_image)
cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
cnn = AveragePooling2D((2,2))(cnn)
cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
cnn = GlobalAveragePooling2D()(cnn)
dense = Dense(128, activation='relu')(cnn)
output = Dense(10, activation='sigmoid')(dense)

model = Model(inputs=input_image, outputs=output)
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test))

Y_pred = model.predict(X_test) #用模型进行预测
greater = np.sort(Y_pred, axis=1)[:,-2] > 0.5 #判断预测结果是否大于0.5
Y_pred = Y_pred.argsort()[:,-2:] #取最高分数的两个类别
Y_pred.sort(axis=1) #排序，因为只比较集合

acc = 1.*(np.prod(Y_pred == Y_test, axis=1)).sum()/len(X_test)
print(u'CNN+Pooling，不考虑置信度的准确率为：%s'%acc)
acc = 1.*(np.prod(Y_pred == Y_test, axis=1)*greater).sum()/len(X_test)
print(u'CNN+Pooling，考虑置信度的准确率为：%s'%acc)

#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#input_1 (InputLayer)         (None, None, None, 1)     0         
#_________________________________________________________________
#conv2d_1 (Conv2D)            (None, None, None, 64)    640       
#_________________________________________________________________
#conv2d_2 (Conv2D)            (None, None, None, 64)    36928     
#_________________________________________________________________
#average_pooling2d_1 (Average (None, None, None, 64)    0         
#_________________________________________________________________
#conv2d_3 (Conv2D)            (None, None, None, 128)   73856     
#_________________________________________________________________
#conv2d_4 (Conv2D)            (None, None, None, 128)   147584    
#_________________________________________________________________
#global_average_pooling2d_1 ( (None, 128)               0         
#_________________________________________________________________
#dense_1 (Dense)              (None, 128)               16512     
#_________________________________________________________________
#dense_2 (Dense)              (None, 10)                1290      
#=================================================================
#Total params: 276,810
#Trainable params: 276,810
#Non-trainable params: 0
#_________________________________________________________________
#Train on 60000 samples, validate on 10000 samples
#


#搭建CNN+Capsule分类模型
input_image = Input(shape=(None,None,1))
cnn = Conv2D(64, (3, 3), activation='relu')(input_image)
cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
cnn = AveragePooling2D((2,2))(cnn)
cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
cnn = Reshape((-1, 128))(cnn)
capsule = Capsule(10, 16, 3, True)(cnn)
output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(10,))(capsule)

model = Model(inputs=input_image, outputs=output)
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test))

Y_pred = model.predict(X_test) #用模型进行预测
greater = np.sort(Y_pred, axis=1)[:,-2] > 0.5 #判断预测结果是否大于0.5
Y_pred = Y_pred.argsort()[:,-2:] #取最高分数的两个类别
Y_pred.sort(axis=1) #排序，因为只比较集合

acc = 1.*(np.prod(Y_pred == Y_test, axis=1)).sum()/len(X_test)
print(u'CNN+Capsule，不考虑置信度的准确率为：%s'%acc)
acc = 1.*(np.prod(Y_pred == Y_test, axis=1)*greater).sum()/len(X_test)
print(u'CNN+Capsule，考虑置信度的准确率为：%s'%acc)

#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#input_1 (InputLayer)         (None, None, None, 1)     0         
#_________________________________________________________________
#conv2d_1 (Conv2D)            (None, None, None, 64)    640       
#_________________________________________________________________
#conv2d_2 (Conv2D)            (None, None, None, 64)    36928     
#_________________________________________________________________
#average_pooling2d_1 (Average (None, None, None, 64)    0         
#_________________________________________________________________
#conv2d_3 (Conv2D)            (None, None, None, 128)   73856     
#_________________________________________________________________
#conv2d_4 (Conv2D)            (None, None, None, 128)   147584    
#_________________________________________________________________
#reshape_1 (Reshape)          (None, None, 128)         0         
#_________________________________________________________________
#capsule_1 (Capsule)          (None, 10, 16)            20480     
#_________________________________________________________________
#lambda_1 (Lambda)            (None, 10)                0         
#=================================================================
#Total params: 279,488
#Trainable params: 279,488
#Non-trainable params: 0
#_________________________________________________________________
#Train on 60000 samples, validate on 10000 samples

