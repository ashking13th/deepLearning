import cv2
import numpy as np
import random
import os, argparse
import tensorflow as tf
import pandas as pd
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
import math

# Conv1D(filters=32, kernel_size=3, strides=1, padding='causal', dilation_rate=2**i, activation='', kernel_regularizer='')(X)

# dilation = 1
filters = 32
kernelSize = 3
dilationDepth = 11

def waveModel(inputShape):

    def residualBlock(orginalX, dilationRate):

        # will act like a gated activation unit
        tanhOut = Conv1D(filters=filters, kernel_size=kernelSize, padding='causal', dilation_rate=dilationRate, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(orginalX)
        sigmOut = Conv1D(filters=filters, kernel_size=kernelSize, padding='causal', dilation_rate=dilationRate, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))(orginalX)

        resLayer = Multiply()([tanhOut, sigmOut])
        resLayer = Conv1D(filters=filters, kernel_size=1, padding='same', kernel_regularizer=regularizers.l2(0.001))(resLayer)
        resLayer = Dropout(0.5)(resLayer)

        skipLayer = resLayer
        resLayer = Add()([orginalX, resLayer])

        return resLayer, skipLayer



    inp = Input(inputShape)

    # wavenet based Model
    # Firstly a causal conv layer on the input
    out = Conv1D(filters=filters, kernel_size=kernelSize, padding='causal')(inp)

    skipLayers = []
    for i in range(dilationDepth + 1):
        out, skipOut = residualBlock(out, 2**i)
        skipLayers.append(skipOut)

    out = Add()(skipLayers)
    out = Activation('relu')(out)
    out = Conv1D(filters=filters, kernel_size=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(out)
    out = Conv1D(filters=1, kernel_size=1, padding='same')(out)

    model = Model(inputs=[inp], outputs=[out])
    return model
    # now residual blocks and skip connections



train_df = pd.read_csv('/home/dlagroup13/wd/project/project_data/train.csv', dtype={'acoustic_data': np.int8, 'time_to_failure': np.float32})

x_train = train_df.acoustic_data.values
y_train = train_df.time_to_failure.values

print(len(x_train))
print(len(y_train))

import gc
del train_df
gc.collect()

# ends_mask = np.less(y_train[:-1], y_train[1:])
# segment_ends = np.nonzero(ends_mask)

# train_seg = []
# start = 0
# for end in segment_ends[0]:
#     train_seg.append((start, end))
#     start = end

# Data = [i for i in range(16)]
# trainData = random.sample(range(16), 14)
# valData = [i for i in Data if i not in trainData]

# if this is changed please change the generator function accordingly
# while the different batch sizes is not possible because all have different sizes of the segment

bins = 135000
binSize = 4096              # time steps
rng = bins*binSize
lst = len(x_train)

stepsTrain = math.ceil(rng / binSize)
stepsVal = math.ceil((lst - rng) / binSize)

xTrain = np.empty((bins, binSize))
yTrain = np.empty((bins, binSize))

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    # denom[denom==0] = 1
    if denom == 0:
        denom = 1
    return x_min + nom/denom

for rw in range(bins):
    for col in range(binSize):
        xTrain[rw,col] = x_train[rw * binSize + col]
        yTrain[rw,col] = y_train[rw * binSize + col]
    xTrain[rw] = scale(xTrain[rw], 0, 1)

x_train = x_train[rng:]
y_train = y_train[rng:]

rng_state = np.random.get_state()
np.random.shuffle(xTrain)
np.random.set_state(rng_state)
np.random.shuffle(yTrain)



# yield np.expand_dims(np.expand_dims(np.array(inpts), axis=0), axis=2),\
# np.expand_dims(np.expand_dims(np.array(outps), axis=0), axis=2)

def generator():
    while True:
        for i in range(bins):
            yield np.expand_dims(np.expand_dims(np.array(xTrain[i]), axis=0), axis=2),\
            np.expand_dims(np.expand_dims(np.array(yTrain[i]), axis=0), axis=2)


def valGenerator():
    while True:  
        sz = 0
        inpts = []
        outps = []
        for i in range(lst-rng):
            inpts.append(x_train[i])
            outps.append(y_train[i])
            sz += 1
            if (i == lst-rng-1) or (sz == binSize):
                sz = 0
                yield np.expand_dims(np.expand_dims(scale(np.array(inpts), 0, 1), axis=0), axis=2),\
                np.expand_dims(np.expand_dims(scale(np.array(outps), 0, 1), axis=0), axis=2)
                inpts = []
                outps = []



model = waveModel((None,1))
print(model.summary())
from tensorflow.keras.optimizers import *
optimizer = Adam(lr=0.0005,decay=1e-6)
model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_absolute_error'])

filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(generator(), steps_per_epoch=stepsTrain, epochs=5, validation_data=valGenerator(), validation_steps=stepsVal, callbacks=callbacks_list)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)