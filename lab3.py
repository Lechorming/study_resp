import torch
import numpy as np
from matplotlib import pyplot as plt
import cv2


#a)对包含矩形对象的二进制图像进行膨胀操作。
BW=np.zeros((900,1000))
BW[400:600,400:700]=1
cv2.imshow('img1',BW)
cv2.waitKey(delay = 0)
# se=strel('square',3);
k = np.ones((300, 300))
# for i in range(50):
#     k[i,i]=1
# cv2.imshow('img2',k)
# cv2.waitKey(delay = 0)
BW2=cv2.dilate(BW,k)
cv2.imshow('img3',BW2)
cv2.waitKey(delay = 0)

#b)改变上述结构元素类型(如：line, diamond, disk等)，重新进行膨胀操作。
# BW=zeros(9,10);
# BW(4:6,4:7)=1;
# imshow(BW,'notruesize')
# se=strel('line',3,3); 		
# BW2=imdilate(BW,se);
# figure,imshow(BW2,'notruesize')