import cv2
import numpy as np
import os

def getImageVariable(path):
	classFolder=path
	# imagePath="/home/bharat/DL/assignment1/classes"
	imageList=[]
	cnt=0
	tempArry=np.zeros((96000,5),int)

	for length in range(2):
		for width in range(2):
			for angle in range(12):
				for color in range(2):

					for variation in range(1000):
	
						tempArry[cnt,:]=[cnt,length,width,angle,color]
						cnt+=1
	np.random.shuffle(tempArry)

	
	for i in range(96000):
		string="/"+str(tempArry[i][1])+"_"+str(tempArry[i][2])+"_"+str(tempArry[i][3])+"_"+str(tempArry[i][4])
		imgFolder=classFolder+string
		img=imgFolder+string+"_"+str(tempArry[i][0]%1000)+".jpeg"
		imageList.append(cv2.imread(img))

	return imageList,tempArry[:,1],tempArry[:,2],tempArry[:,3],tempArry[:,4]


imagePath="/home/bharat/DL/assignment1/classes" # jaha imagees ka folder h
a,b,c,d,e=getImageVariable(imagePath)
print(a[0],b[0],c[0],d[0],e[0])