import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

color1=np.array([0,0,255],np.uint8)
color2=np.array([255,0,0],np.uint8)
colors=[color1,color2]
lengths=np.array([7,15],np.uint8)
widths=np.array([1,3],np.uint8)

topLeft=np.zeros(2,int)
bottomRight=np.zeros(2,int)
os.mkdir("classes")
path=os.getcwd()
folderpath=os.path.join(path,"classes")

def topLeftCorner(x):
	mxX=-1
	mxY=-1
	for row in range(28):
		for col in range(28):
			if(np.sum(x[row,col])>0):
				if(abs(14-row)>mxY):
					mxY=abs(14-row)
				if(abs(14-col)>mxX):
					mxX=abs(14-col)
	return np.array([mxY,mxX],int)
				

for length in range(2):
	for width in range(2):
		for angle in range(12):
			for color in range(2):
				img=np.zeros((28,28,3),np.uint8)
				l1=int(14-lengths[length]/2)
				l2=int(15+lengths[length]/2)
				w1=int(14-widths[width]/2)
				w2=int(15+widths[width]/2)
				img[w1:w2,l1:l2]=colors[color]
				rotateAngle=angle*15
				rotation_matrix = cv2.getRotationMatrix2D((14, 14),rotateAngle, 1)
				img_rotated = cv2.warpAffine(img, rotation_matrix, (28,28))
				corner=topLeftCorner(img_rotated)
				vertical=corner[0]+3
				horizontal=corner[1]+3
				string=str(length)+"_"+str(width)+"_"+str(angle)+"_"+str(color)

				fpath=folderpath+"/"+string
				# if(os.path.exists(fpath)):
				# 	os.rmdirs(fpath)
				os.mkdir(fpath)
				variation=0

				
				while variation<1000:
					for i in range(vertical,27-vertical,1):
						for j in range(horizontal,27-horizontal,1):
							translation_matrix=np.float32([ [1,0,j-14], [0,1,i-14] ])
							tranImage=cv2.warpAffine(img_rotated, translation_matrix, (28,28))
							iName=str(length)+"_"+str(width)+"_"+str(angle)+"_"+str(color)+"_"+str(variation)+".jpeg"
							file=fpath+"/"+iName
							# plt.imshow(tranImage)
							# plt.savefig(file)
							cv2.imwrite(file,tranImage)
							variation+=1
							if(variation>=1000):
								break;
						if(variation>=1000):
							break;





