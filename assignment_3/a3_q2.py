import tensorflow as tf
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from PIL import Image
# import os

# Data prep
raw_ip_folder = "/wd/users/dlagroup13/Assignment3/Q2/"

img_pre = "Image_original_"
mask_pre = "_groundtruth_(1)_Image_"

# images
count = 0
im_data = []
mask_data = []

print("Meow")
# for root, dirs, files in os.walk(raw_ip_folder+"Data/"):
#     for f in files:
#         count += 1
#         if(count % 500 == 0):
#             print("count : ",count)
#         im_path = raw_ip_folder + "Data/" + f
#         mask_path = raw_ip_folder + "Mask/" + mask_pre + f[15:]
        
#         im_data.append(cv2.imread(im_path))
#         mask_data.append(cv2.imread(mask_path))

# im_data = np.array(im_data)
# mask_data = np.array(mask_data)

# print("image data size : ", im_data.shape[0])
# print("image size : ", im_data[0].shape)
# print("mask data size : ", mask_data.shape[0])
# print("mask size : ", mask_data[0].shape)

# split_size = int(im_data.shape[0]*0.7)

# im_data_train = im_data[:split_size]
# im_data_test = im_data[split_size:]
# mask_data_train = mask_data[:split_size]
# mask_data_test = mask_data[split_size:]

# print("image data train size : ", im_data_train.shape)
# print("image data test size : ", im_data_test.shape)
# print("mask data train size : ", mask_data_train.shape)
# print("mask data test size : ", mask_data_test.shape)

# np.save("/wd/users/dlagroup13/prep_data/im_train", im_data_train)
# np.save("/wd/users/dlagroup13/prep_data/im_test", im_data_test)
# np.save("/wd/users/dlagroup13/prep_data/mask_train", mask_data_train)
# np.save("/wd/users/dlagroup13/prep_data/mask_test", mask_data_test)