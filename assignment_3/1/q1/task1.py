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

folpath="/home/test/question1/Q1/task1/Data"
# folpath="/home/bharat/DL/assignment3/question1/Q1/task1/Data"
imshape=(352,288)
dataArray=[]
optArray=[]

knuckle=folpath+"/Knuckle"
dt=knuckle+"/Data"
fl=knuckle+"/groundtruth.txt"
# print(fl)
f=open(fl,"r")
inameArray=["Data/"+x for x in os.listdir(dt)]

for x in f:
  y=x.split(",")
  p,q=y[0].split(" ")
  p=p+"_"+q
  impath=knuckle+"/"+p
#   print(p)
  if p in inameArray:
    img=cv2.imread(impath)
#     cv2.imshow("img",img)
    rimg=cv2.resize(img,imshape)
    rimg=rimg.astype(float)
    rimg/=255.0
    sp=img.shape
    rt2=288.0/float(sp[0])
    rt1=352.0/float(sp[1])
#     print(rt1,rt2)
    tp=np.array([0,y[1],y[2],y[3],y[4]],float)
    ar=np.array([0,tp[1]*rt1,tp[2]*rt2,tp[3]*rt1,tp[4]*rt2],int)
    dataArray.append(rimg)
    optArray.append(ar)

palm=folpath+"/Palm"
dt=palm+"/Data"
fl=palm+"/groundtruth.txt"
# print(fl)
f=open(fl,"r")
inameArray=["Data/"+x for x in os.listdir(dt)]

for x in f:
  y=x.split(",")
  impath=palm+"/"+y[0]
#   print(y[0])
  if y[0] in inameArray:
    img=cv2.imread(impath)
    rimg=cv2.resize(img,imshape)
    rimg=rimg.astype(float)
    rimg/=255.0
    sp=img.shape
    rt2=288.0/float(sp[0])
    rt1=352.0/float(sp[1])
#     print(rt1,rt2)
    tp=np.array([1,y[1],y[2],y[3],y[4]],float)
    ar=np.array([1,tp[1]*rt1,tp[2]*rt2,tp[3]*rt1,tp[4]*rt2],int)
#     print(rimg.shape)
    dataArray.append(rimg)
    optArray.append(ar)
    
vein=folpath+"/Vein"
dt=vein+"/Data"
fl=vein+"/groundtruth.txt"
# print(fl)
f=open(fl,"r")
inameArray=["Data/"+x for x in os.listdir(dt)]
# print(inameArray)
for x in f:
  y=x.split(",")
  p,q=y[0].split("/0")
  p=p+"/"+q
  impath=vein+"/"+p
#   print(y[0])
  if p in inameArray:
    img=cv2.imread(impath)
    rimg=cv2.resize(img,imshape)
    rimg=rimg.astype(float)
    rimg/=255.0
    sp=img.shape
    rt2=288.0/float(sp[0])
    rt1=352.0/float(sp[1])
#     print(rt1,rt2)
    tp=np.array([2,y[1],y[2],y[3],y[4]],float)
    ar=np.array([2,tp[1]*rt1,tp[2]*rt2,tp[3]*rt1,tp[4]*rt2],int)
    dataArray.append(rimg)
    optArray.append(ar)
  
iptdata=np.array(dataArray)
optdata=np.array(optArray)

X_train, X_test, y_train, y_test = train_test_split(iptdata, optdata, test_size=0.2)
opt1_train=to_categorical(y_train[:,0])
opt1_test=to_categorical(y_test[:,0])

# print(X_train.shape,len(dataArray),y_test.shape)

inp = Input(shape=(288,352,3))

# Conv -> Conv -> Pool
x = Conv2D(32, (7,7), padding="same")(inp)

x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Conv2D(32, (7,7), padding="same")(x)

x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2))(x)


# Conv -> Conv -> Pool
x = Conv2D(32, (5,5), padding="same")(x)

x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Conv2D(32, (5,5), padding="same")(x)

x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Flatten()(x)

# above is the shared layer and our x now
# is the feature map.
# x = (7, 7, 64)

# out1 corresponds to the length clasification head
out1 = Dense(30)(x)
out1 = Activation("relu")(out1)
out1 = Dropout(0.1)(out1)

out1 = Dense(10)(x)
out1 = Activation("relu")(out1)
# out1 = Dense(1024)(out1)
# out1 = Activation("relu")(out1)
# out1 = Dropout(0.3)(out1)
out1 = Dense(3)(out1)
out1 = Activation("sigmoid", name="out1")(out1)

# out2 corresponds to the width clasification head
out2 = Dense(40)(x)
out2 = Activation("relu")(out2)
out2 = Dropout(0.1)(out2)

out2 = Dense(20)(x)
out2 = Activation("relu")(out2)
# out2 = Dense(1024)(out2)
# out2 = Activation("relu")(out2)
# out2 = Dropout(0.5)(out2)
out2 = Dense(4)(out2)
out2 = Activation("linear", name="out2")(out2)



# final model with one input and four classification heads
model = Model(inputs=[inp], outputs=[out1, out2])

model.summary()
# plot_model(model, to_file='assign2b.png')

model.compile(optimizer='adam', loss={'out1': 'categorical_crossentropy', 'out2': 'mean_squared_error'}, metrics=['accuracy'])
infoFit= model.fit([X_train], {'out1': opt1_train, 'out2': y_train[:,1:5]}, epochs=10, batch_size=128)

result = model.predict(X_test)
score = model.evaluate(X_test, {'out1': opt1_test, 'out2': y_test[:,1:5]})
print(model.metrics_names)
# print(result[3])

# y_test[:,1:5]

print(result)

print(score)
resultSave = np.array(result[1])
print(np.sum(resultSave, axis=1))