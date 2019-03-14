import cv2
import numpy as np
import os
import glob

os.mkdir("vimage")
path=os.getcwd()
folderpath=os.path.join(path,"classes")
vidpath=os.path.join(path,"vimage")
cnt=0
for length in range(2):
	for width in range(2):
		for angle in range(12):
			for color in range(2):

				string=str(length)+"_"+str(width)+"_"+str(angle)+"_"+str(color)
				fpath=folderpath+"/"+string

				ipath=fpath+"/"+str(length)+"_"+str(width)+"_"+str(angle)+"_"+str(color)

				for image in range(10):
					iName=ipath+"_"+str(image*3)+".jpeg"
					im1=cv2.imread(iName)
					iName=ipath+"_"+str(image*3+1)+".jpeg"
					im1=np.concatenate((im1,cv2.imread(iName)),axis=0)
					iName=ipath+"_"+str(image*3+2)+".jpeg"
					im1=np.concatenate((im1,cv2.imread(iName)),axis=0)

					iName=ipath+"_"+str(image*3+3)+".jpeg"
					im2=cv2.imread(iName)
					iName=ipath+"_"+str(image*3+4)+".jpeg"
					im2=np.concatenate((im2,cv2.imread(iName)),axis=0)
					iName=ipath+"_"+str(image*3+5)+".jpeg"
					im2=np.concatenate((im2,cv2.imread(iName)),axis=0)

					iName=ipath+"_"+str(image*3+6)+".jpeg"
					im3=cv2.imread(iName)
					iName=ipath+"_"+str(image*3+7)+".jpeg"
					im3=np.concatenate((im3,cv2.imread(iName)),axis=0)
					iName=ipath+"_"+str(image*3+8)+".jpeg"
					im3=np.concatenate((im3,cv2.imread(iName)),axis=0)

					im=np.concatenate((im1,im2),axis=1)
					im=np.concatenate((im,im3),axis=1)

					p1=vidpath+"/"+str(cnt)+".jpeg"
					cv2.imwrite(p1,im)
					cnt+=1

 
img_array = []
for fileNo in range(960):
	filepath=vidpath+"/"+str(fileNo)+".jpeg"
	img = cv2.imread(filepath)
	height, width, layers = img.shape
	size = (width,height)
	img_array.append(img)

 
out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()