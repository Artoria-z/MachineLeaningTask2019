# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 09:33:23 2019

@author: 99628
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:36:59 2019

@author: 99628
"""
import pandas as pd     
import numpy as np
import keras
import os
import os.path
os.environ['KERAS_BACKEND']='tensorflow'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D,MaxPooling2D,BatchNormalization,Activation
from keras.optimizers import Adam
from matplotlib import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from mylib.models import densesharp, metrics, losses,DenseNet
from mylib.models.DenseNet import createDenseNet
from keras.optimizers import SGD
import pandas as pd
from keras.losses import categorical_crossentropy

from keras.callbacks import EarlyStopping

#########数据总数465 分出训练集的样本数为4/5 1/5为test集  留出法
x_path='./train_val'
x_file=os.listdir(x_path)
x_filef_train=x_file[0:373]
x_filef_test=x_file[373:465]
x_test_path='./test'

def get_dataset():
    x_return_train=[]
    x_return_test=[]
    x_name=pd.read_csv("train_val.csv") ['name']
    for i in range(len(x_filef_train)):
        x_file_temp=os.path.join(x_path,x_name[i]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel
        x_return_train.append(x_temp[34:66,34:66,34:66])
    for i in range(len(x_filef_test)):
        x_file_temp=os.path.join(x_path,x_name[i+373]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel
        x_return_test.append(x_temp[34:66,34:66,34:66])
    return  x_return_train,x_return_test

def get_label():
    x_label=pd.read_csv("train_val.csv") ['lable']
    x_train_label=keras.utils.to_categorical(x_label,2)[0:373]
    x_test_label=keras.utils.to_categorical(x_label,2)[373:465]
    return x_train_label,x_test_label

def get_testdataset():
    x_return=[]
    x_name=pd.read_csv("test_2.csv") ['Id']
    for i in range(117):
        x_file_temp=os.path.join(x_test_path,x_name[i]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel
        x_return.append(x_temp[34:66,34:66,34:66])
    return x_return

def get_batch(x, y, step, batch_size, alpha=0.2):
    """
    get batch data
    :param x: training data
    :param y: one-hot label
    :param step: step
    :param batch_size: batch size
    :param alpha: hyper-parameter α, default as 0.2
    :return:
    """
    candidates_data, candidates_label = x, y
    offset = (step * batch_size) % (candidates_data.shape[0] - batch_size)

    # get batch data
    train_features_batch = candidates_data[offset:(offset + batch_size)]
    train_labels_batch = candidates_label[offset:(offset + batch_size)]

    # 最原始的训练方式
    if alpha == 0:
        return train_features_batch, train_labels_batch
    # mixup增强后的训练方式
    if alpha > 0:
        weight = np.random.beta(alpha, alpha, batch_size)
        x_weight = np.zeros((batch_size, 32,32,32,1))
        for jj in range(batch_size):
            for hh in range(32):
                for gg in range(32):
                    for tt in range(32):
                        x_weight[jj,hh,gg,tt,0] = weight[jj]
        y_weight = weight.reshape(batch_size, 1)
        index = np.random.permutation(batch_size)
        y1, y2 = train_labels_batch, train_labels_batch[index]
        y = y1 * y_weight + y2 * (1 - y_weight)
        x1, x2 = train_features_batch, train_features_batch[index]
        x = x1* x_weight + x2* (1 - x_weight)
        return x, y

densenet_depth =40
densenet_growth_rate = 20
batch_size = 32
x_train,x_test=get_dataset()
x_train=np.array(x_train)
x_test=np.array(x_test)
x_train_label,x_test_label=get_label()
x_train=x_train.reshape(x_train.shape[0],32,32,32,1)
x_train=x_train.astype('float32')/255
x_test=x_test.reshape(x_test.shape[0],32,32,32,1)
x_test=x_test.astype('float32')/255

x_train,x_train_label=get_batch(x_train,x_train_label,1,1,0.2)
x_test, x_test_label=get_batch(x_test,x_test_label,1,1,0.2)
x_predict=np.array(get_testdataset())
x_predict=x_predict.reshape(x_predict.shape[0],32,32,32,1)
x_predict=x_predict.astype('float32')/255

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
nb_classes = 2

model = createDenseNet(nb_classes=nb_classes,img_dim=[32,32,32,1],depth=densenet_depth,growth_rate = densenet_growth_rate)
model.compile(loss=categorical_crossentropy,optimizer=Adam(), metrics=['accuracy'])
model.summary()  # print the model

model.fit(x_train, x_train_label,batch_size=8, epochs=15,validation_data=(x_test, x_test_label), verbose=2, shuffle=False, callbacks=[early_stopping])
loss,accuracy = model.evaluate(x_train,x_train_label)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (loss,accuracy))
loss,accuracy = model.evaluate(x_test,x_test_label)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (loss,accuracy))
####最终估计的地方
#print(model.predict(x_predict))
model.save("model.h5")
y_pred = model.predict(x_predict, batch_size=1)
np.savetxt('test.csv', y_pred, delimiter=',')