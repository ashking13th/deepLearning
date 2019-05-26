# -*- coding: utf-8 -*

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

folpath="/home/dlagroup13/test/question1/Four_Slap_Fingerprint"
# folpath="/home/bharat/DL/assignment3/question1/Q1/task1/Data"/home/bharat/DL/assignment3/question1/Four_Slap_Fingerprint
imshape=(1672,1572)
dataArray=[]
optArray=[]

dtpath=folpath+"/Data"
gtpath=folpath+"/Ground_truth"
# print(fl)
inameArray=[x for x in os.listdir(dtpath)]

for gttxt in os.listdir(gtpath):
  a,b=gttxt.split(".tx")
  iName=a+".jpg"
  if iName in inameArray:
    imgpath=dtpath+"/"+iName
    img=cv2.imread(imgpath)
    rimg=cv2.resize(img,imshape)
    rimg=rimg.astype(float)
    rimg/=255.0
    sp=img.shape
    rt2=1572.0/float(sp[0])
    rt1=1672.0/float(sp[1])

    txtfile=gtpath+"/"+gttxt
    f=open(txtfile,"r")
    lst=[]
    for line in f:
      a,b,c,d=line.split(",")
      lst.append(float(a)*rt2)
      lst.append(float(b)*rt1)
      lst.append(float(c)*rt2)
      lst.append(float(d)*rt1)
    ar=np.array(lst,int)
    dataArray.append(rimg)
    optArray.append(ar)

  
iptdata=np.array(dataArray)
optdata=np.array(optArray)

X_train, X_test, y_train, y_test = train_test_split(iptdata, optdata, test_size=0.2)

print(X_train.shape,len(dataArray),y_test.shape)

inp = Input(shape=(1572,1672,3))

# Conv -> Conv -> Pool
x = Conv2D(32, (7,7), padding="same")(inp)

x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Conv2D(32, (7,7), padding="same")(x)

x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(8,8))(x)


# Conv -> Conv -> Pool
# x = Conv2D(32, (10,10), padding="same")(x)

# x = Activation("relu")(x)
# x = BatchNormalization()(x)
x = Conv2D(32, (5,5), padding="same")(x)

x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(8,8))(x)
x = Flatten()(x)

# above is the shared layer and our x now
# is the feature map.
# x = (7, 7, 64)

out2 = Dense(100)(x)
out2 = Activation("relu")(out2)
out2 = Dropout(0.1)(out2)

out2 = Dense(50)(x)
out2 = Activation("relu")(out2)

# out2 = Dense(20)(x)
# out2 = Activation("relu")(out2)
# out2 = Dropout(0.2)(out2)

out2 = Dense(20)(x)
out2 = Activation("relu")(out2)
# out2 = Dense(1024)(out2)
# out2 = Activation("relu")(out2)
# out2 = Dropout(0.5)(out2)
out2 = Dense(16)(out2)
out2 = Activation("linear", name="out2")(out2)



# final model with one input and four classification heads
model = Model(inputs=[inp], outputs=[out2])

model.summary()
# plot_model(model, to_file='assign2b.png')

model.compile(optimizer='adam', loss={'out2': 'mean_squared_error'}, metrics=['accuracy'])
infoFit= model.fit([X_train], {'out2': y_train}, epochs=10, batch_size=64)

result = model.predict(X_test)
score = model.evaluate(X_test, {'out2': y_test})
print(model.metrics_names)
# print(result[3])

# y_test[:,1:5]

print(result)

print(score)
resultSave = np.array(result[1])
print(np.sum(resultSave, axis=1))