from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility
import tensorflow as tf
import tensorflow.keras.models as models
import os
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras import Input
import cv2

data_path = "/home/dlagroup13/wd/data_extract/"

x_train = np.load(data_path + "x_train.npy", mmap_mode='r')[:4000]/255.0
print(x_train.shape)
y_train = np.load(data_path + "y_train.npy", mmap_mode='r')[:4000]/255.0
print(y_train.shape)

def unet(pretrained_weights = None,input_size = (300,400,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print("Check P 1 ", pool1.shape)
    
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print("Check P 2 ", pool2.shape)
    
    conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    drop5 = Dropout(0.5)(conv5)
    print("Check 2 ", drop5.shape)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    print("Check 5 ", conv8.shape)

    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = [inputs], outputs = [conv10])
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

model = unet()

model.fit(x_train, y_train, batch_size = 16 ,epochs=8, validation_split = 0.1)

model.save_weights('/home/dlagroup13/wd/unet_03.hdf5')
model.save('/home/dlagroup13/wd/unet_03.h5')

data_path = "/content/drive/My Drive/data_extract/"

x_test = np.load(data_path + "x_test.npy", mmap_mode='r')[:400]/255.0
print(x_test.shape)
y_test = np.load(data_path + "y_test.npy", mmap_mode='r')[:400]/255.0
print(y_test.shape)

results = autoencoder.predict_proba(x_test)
score = autoencoder.evaluate(x_test, y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print(score)
print(results[0].shape)
print(results.shape)

np.save("results", results)