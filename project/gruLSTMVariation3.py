# import numpy as np 
# import pandas as pd
# import os
# from tqdm import tqdm
# from keras.models import *
# from keras.layers import *
# from keras.callbacks import *
# from keras.initializers import *
# from keras.optimizers import adam

# pd.set_option('precision', 30)
# np.set_printoptions(precision = 30)

# float_data = pd.read_csv("/home/dlagroup13/wd/project/project_data/train.csv", dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}).values
# inputTrain = float_data[:,0]
# outputTrain = float_data[:,1]

# import gc
# del float_data
# gc.collect()

# lahead = 1
# batchSize = 10
# epochs = 20

# # last = inputTrain.shape[0] % batchSize;
# inputTrain = inputTrain[:-5480]
# outputTrain = outputTrain[:-5480]

# from sklearn import preprocessing

# inputTrain = inputTrain.reshape(inputTrain.shape[0],1)
# inputTrain = preprocessing.normalize(inputTrain)

# inputTrain = inputTrain.reshape((629140,1000,1))
# # outputTrain = outputTrain.reshape((629140,1,1))

# y_train = np.empty((629140,1))
# start = -1

# for i in range(629140):
# 	y_train[i] = outputTrain[start + 1000]
# 	start += 1000

# del outputTrain
# gc.collect()

# y_train = y_train.reshape((629140,1,1))

# model = Sequential()
# model.add(Conv1D(filters=16,kernel_size=10,strides=10,padding='valid',activation='relu', batch_size=10,input_shape=(1000,1)))
# model.add(Conv1D(filters=32,kernel_size=10,strides=10,padding='valid',activation='relu'))
# model.add(Conv1D(filters=64,kernel_size=10,strides=10,padding='valid',activation='relu'))
# model.add(CuDNNGRU(70, stateful=True, return_sequences=True))
# model.add(CuDNNGRU(40, stateful=True, return_sequences=True))
# model.add(TimeDistributed(Dense(20)))
# model.add(TimeDistributed(Dense(10)))
# model.add(TimeDistributed(Dense(1)))

# model.compile(loss='mse', optimizer='adam')
# model.summary()

# for i in range(epochs):
#     print('Epoch', i + 1, '/', epochs)
#     # Note that the last state for sample i in a batch will
#     # be used as initial state for sample i in the next batch.
#     # Thus we are simultaneously training on batch_size series with
#     # lower resolution than the original series contained in data_input.
#     # Each of these series are offset by one step and can be
#     # extracted with data_input[i::batch_size].
#     model.fit(inputTrain,
#                y_train,
#                batch_size=10,
#                epochs=1,
#                verbose=1,
#                shuffle=False)
#     model.save("model_" + str(i) + ".h5")
#     model.reset_states()	


# BASIC IDEA OF THE KERNEL

# The data consists of a one dimensional time series x with 600 Mio data points. 
# At test time, we will see a time series of length 150'000 to predict the next earthquake.
# The idea of this kernel is to randomly sample chunks of length 150'000 from x, derive some
# features and use them to update weights of a recurrent neural net with 150'000 / 1000 = 150
# time steps. 

import numpy as np 
import pandas as pd
import os
from tqdm import tqdm
from scipy.stats import moment

# Fix seeds
from numpy.random import seed
seed(639)
from tensorflow import set_random_seed
set_random_seed(5944)

# Import
float_data = pd.read_csv("/home/dlagroup13/wd/project/project_data/train.csv", dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}).values

# Helper function for the data generator. Extracts some moments of data.
# Can easily be extended. Expects a two dimensional array.
def extract_features(z):
	lt = z.mean(axis=1)
	for i in range(2,11):
		lt = np.c_[lt, moment(z, axis=1, moment=i)]
	return lt


# For a given ending position "last_index", we split the last 150'000 values 
# of "x" into 150 pieces of length 1000 each. So n_steps * step_length should equal 150'000.
# From each piece, a set features are extracted. This results in a feature matrix 
# of dimension (150 time steps x features).  
def create_X(x, last_index=None, n_steps=150, step_length=1000):
    if last_index == None:
        last_index=len(x)
       
    assert last_index - n_steps * step_length >= 0

    # Reshaping and approximate standardization with mean 5 and std 3.
    temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5 ) / 3
    
    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 
    # of the last 10 observations. 
    return np.c_[extract_features(temp),
                 extract_features(temp[:, -step_length // 10:]),
                 extract_features(temp[:, -step_length // 100:])]

# Query "create_X" to figure out the number of features
n_features = create_X(float_data[0:150000]).shape[1]
print("Our RNN is based on %i features"% n_features)
    
# The generator endlessly selects "batch_size" ending positions of sub-time series. For each ending position,
# the "time_to_failure" serves as target, while the features are created by the function "create_X".
def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):
    if max_index is None:
        max_index = len(data) - 1
     
    while True:
        # Pick indices of ending positions
        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)
         
        # Initialize feature matrices and targets
        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size, )
        
        for j, row in enumerate(rows):
            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)
            targets[j] = data[row - 1, 1]
        yield samples, targets
        
batch_size = 1

# Position of second (of 16) earthquake. Used to have a clean split
# between train and validation
second_earthquake = 50085877
# float_data[second_earthquake, 1]

# Initialize generators
train_gen = generator(float_data, batch_size=batch_size) # Use this for better score
# train_gen = generator(float_data, batch_size=batch_size, min_index=second_earthquake + 1)
valid_gen = generator(float_data, batch_size=batch_size, max_index=second_earthquake)

# Define model
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint

cb = [ModelCheckpoint("model.hdf5", save_best_only=True)]

model = Sequential()
model.add(CuDNNGRU(50, input_shape=(None, n_features), return_sequences=True))
model.add(CuDNNGRU(20, input_shape=(None, n_features)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# Compile and fit model
model.compile(optimizer=adam(lr=0.0005), loss="mae")

history = model.fit_generator(train_gen,
                              steps_per_epoch=10000,
                              epochs=200,
                              callbacks=cb,
                              validation_data=valid_gen,
                              validation_steps=200)

# Visualize accuracies
# import matplotlib.pyplot as plt

# def perf_plot(history, what = 'loss'):
#     x = history.history[what]
#     val_x = history.history['val_' + what]
#     epochs = np.asarray(history.epoch) + 1
    
#     plt.plot(epochs, x, 'bo', label = "Training " + what)
#     plt.plot(epochs, val_x, 'b', label = "Validation " + what)
#     plt.title("Training and validation " + what)
#     plt.xlabel("Epochs")
#     plt.legend()
#     plt.show()
#     return None

# perf_plot(history)

# Load submission file
submission = pd.read_csv('/home/dlagroup13/wd/project/project_data/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(tqdm(submission.index)):
  #  print(i)
    seg = pd.read_csv('/home/dlagroup13/wd/project/project_data/test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values
    submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x), 0))

submission.head()

# Save
submission.to_csv('submission.csv')