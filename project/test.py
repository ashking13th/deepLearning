# import numpy as np 
# import pandas as pd
# import os
# from tqdm import tqdm
# from keras.models import *
# from keras.layers import *
# from keras.callbacks import *
# from keras.initializers import *
# from os import listdir, makedirs
# from os.path import isfile, join, basename, splitext, isfile, exists
# from tqdm import tqdm_notebook

# def load_test(ts_length = 150000):
#     base_dir = '/home/dlagroup13/wd/project/project_data/test/'
#     test_files = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]

#     ts = np.empty([len(test_files), ts_length])
#     ids = []
    
#     i = 0
#     for f in tqdm_notebook(test_files):
#         ids.append(splitext(f)[0])
#         t_df = pd.read_csv(base_dir + f, dtype={"acoustic_data": np.int8})
#         ts[i, :] = t_df['acoustic_data'].values
#         i = i + 1

#     return ts, ids


# # In[20]:


# test_data, test_ids = load_test()
# model = load_model('model_11.h5')
# yPred = []
# from sklearn import preprocessing

# for i in range(test_data.shape[0]):
# 	xTest = test_data[i]
# 	xTest = xTest.reshape(xTest.shape[0], 1)
# 	xTest = preprocessing.normalize(xTest)
# 	xTest = xTest.reshape(150,1000,1)
# 	temp = model.predict(xTest, batch_size=10)
# 	yPred.append(temp[-1])

# yPred = np.array(yPred)
# yPred = np.squeeze(yPred)
# print(yPred.shape)
# submission_df = pd.DataFrame({'seg_id': test_ids, 'time_to_failure': yPred})
# submission_df.to_csv("submission.csv", index=False)

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
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
from keras.optimizers import adam

pd.set_option('precision', 30)
np.set_printoptions(precision = 30)

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


model = load_model('model.hdf5')
submission = pd.read_csv('/home/dlagroup13/wd/project/project_data/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(tqdm(submission.index)):
  #  print(i)
    seg = pd.read_csv('/home/dlagroup13/wd/project/project_data/test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values
    submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x), 0))

submission.head()

# Save
submission.to_csv('submission_0.csv')