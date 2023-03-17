import torch
import numpy
import matplotlib

#a)对包含矩形对象的二进制图像进行膨胀操作。
BW=zeros(9,10);
BW(4:6,4:7)=1;
imshow(BW,'notruesize')
se=strel('square',3);
BW2=imdilate(BW,se);
figure,imshow(BW2,'notruesize')

#b)改变上述结构元素类型(如：line, diamond, disk等)，重新进行膨胀操作。
# BW=zeros(9,10);
# BW(4:6,4:7)=1;
# imshow(BW,'notruesize')
# se=strel('line',3,3); 		
# BW2=imdilate(BW,se);
# figure,imshow(BW2,'notruesize')