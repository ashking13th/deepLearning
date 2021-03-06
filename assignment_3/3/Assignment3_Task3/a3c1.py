# -*- coding: utf-8 -*-
"""A3c.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-s_k48w8mbCDFJUr7J_BNVHRRjiH4js6
"""

# from google.colab import drive
# drive.mount('/content/drive')

# !unzip /content/drive/My\ Drive/Core_Point.zip -d /content/

import cv2
import numpy as np
import random
import os, argparse
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

# take image and corresponding ground truth and plot

from os import listdir
from os.path import isfile, join

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--phase", required=True, help="which phase of the program")
ap.add_argument("-e", "--epochs", help="which phase of the program")
args = vars(ap.parse_args())

if args["phase"] == "train":
  epochs = int(args["epochs"])
  mypath = input("Enter the full path to the training dataset having data and ground truth: ")

  mypath2 = mypath
  mypath = join(mypath, "Data")
  # mypath='/home/dlagroup13/wd/Assignment3/Q3/Core_Point/Data/'
  onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

  # shuffle the onlyfiles list!!
  random.shuffle(onlyfiles)

  # mypath1='/home/dlagroup13/wd/Assignment3/Q3/Core_Point/Ground_truth/'
  mypath1 = join(mypath2, "Ground_truth")

  imgSize = 600

  inputTrain = []
  outputTrain = []

  # for taking in the input in the dictionary corresponding to the shape of the image

  for n in range(0, len(onlyfiles)):
    
    nmp = cv2.imread(join(mypath,onlyfiles[n]), 0)
    nmp = np.array(nmp, dtype=np.float32)
    nmp /= 255.0

    # outliers i.e. the blank images remove them
    if np.array_equal(np.ones(shape=nmp.shape), nmp):
      continue    
           
    height = imgSize - nmp.shape[0]
    width = imgSize - nmp.shape[1]

    # can make a assert so that the image size is limited to 650
    if height > 0:
      t1 = np.zeros((height, nmp.shape[1]))
      nmp = np.concatenate((nmp, t1), axis=0)

    if width > 0:
      t2 = np.zeros((nmp.shape[0], width))
      nmp = np.concatenate((nmp, t2), axis=1)

    if height < 0 or width < 0:
      nmp = cv2.resize(nmp*255.0, (imgSize, imgSize))

    nmp = np.expand_dims(nmp, axis=2)
    
    inputTrain.append(nmp)


    f = open(join(mypath1, onlyfiles[n].split(".")[0] + "_gt.txt"), "r")
    st = f.read().split()
    p = []
    for i in st:
      p.append(int(i))

    outputTrain.append(np.array(p))



  inputTrain = np.array(inputTrain)
  outputTrain = np.array(outputTrain)
  batchSz = 4

  # smooth L1
  HUBER_DELTA = 0.5
  def smoothL1(y_true, y_pred):
    x   = K.abs(y_true - y_pred)
    x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  K.sum(x)

  inp = Input(shape=(imgSize, imgSize, 1))

  x = Conv2D(32, (5,5), dilation_rate=2, padding="valid", activation="relu")(inp)
  x = BatchNormalization()(x)

  x = Conv2D(32, (5,5), dilation_rate=2, padding="valid", activation="relu")(x)
  x = BatchNormalization()(x)

  x = MaxPooling2D(pool_size=(2,2))(x)

  x = Conv2D(64, (5,5), dilation_rate=2, padding="valid", activation="relu")(x)
  x = BatchNormalization()(x)

  x = Conv2D(64, (5,5), dilation_rate=2, padding="valid", activation="relu")(x)
  x = BatchNormalization()(x)

  x = MaxPooling2D(pool_size=(2,2))(x)

  x = Conv2D(64, (3,3), dilation_rate=1, padding="valid", activation="relu")(x)
  x = BatchNormalization()(x)

  x = Conv2D(64, (3,3), dilation_rate=1, padding="valid", activation="relu")(x)
  x = BatchNormalization()(x)

  x = MaxPooling2D(pool_size=(2,2))(x)
  x = GlobalMaxPooling2D()(x)
  x = Flatten()(x)

  x = Dense(1048, activation="relu")(x)

  x = Dropout(0.5)(x)

  x = Dense(256, activation="relu")(x)

  out = Dense(2)(x)

  model = Model(inputs=[inp], outputs=[out])
  model.summary()

  from tensorflow.keras.optimizers import *
  optimizer = RMSprop(lr=1e-3)
  model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

  filepath = "model.h5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
  callbacks_list = [checkpoint]
  model.fit(x=inputTrain, y=outputTrain, batch_size=batchSz, epochs=epochs, validation_split=0.1  , callbacks=callbacks_list)

  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)

else:
  mypath = input("Enter the full path to the test dataset containing images directly: ")
  onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

  inputTest = []
  fileName = []

  imgSize = 600

  # for taking in the input in the dictionary corresponding to the shape of the image

  for n in range(0, len(onlyfiles)):
    
    nmp = cv2.imread(join(mypath,onlyfiles[n]), 0)

    nmp = np.array(nmp, dtype=np.float32)
    nmp /= 255.0

    fileName.append(onlyfiles[n])

    # outliers i.e. the blank images remove them
    # if np.array_equal(np.ones(shape=nmp.shape), nmp):
    #   continue    
           
    height = imgSize - nmp.shape[0]
    width = imgSize - nmp.shape[1]

    # can make a assert so that the image size is limited to 650
    if height > 0:
      t1 = np.zeros((height, nmp.shape[1]))
      nmp = np.concatenate((nmp, t1), axis=0)

    if width > 0:
      t2 = np.zeros((nmp.shape[0], width))
      nmp = np.concatenate((nmp, t2), axis=1)

    if height < 0 or width < 0:
      nmp = cv2.resize(nmp*255.0, (imgSize, imgSize))

    nmp = np.expand_dims(nmp, axis=2)
    
    inputTest.append(nmp)

  json_file = open('model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)

  # load weights into new model
  loaded_model.load_weights("model.h5")  
  from tensorflow.keras.optimizers import *  
  optimizer = RMSprop(lr=1e-3)
  loaded_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])


  inputTest = np.array(inputTest)
  preds = loaded_model.predict(inputTest)

  preds.astype(int)
  path = mypath.rsplit('/', 1)[0]
  path = join(path, "Predictions")
  # print(path)
  if not os.path.exists(path):
      os.makedirs(path)
  for l in range(len(preds)):
    # print(join(path, fileName[l].split(".")[0] + ".txt"))
    np.savetxt(join(path, fileName[l].split(".")[0] + ".txt"), preds[l], newline=" ")