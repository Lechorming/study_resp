import cv2
import numpy as np

a=np.array([
    [3,3,4,6,9],
    [3,15,5,5,7],
    [1,3,4,1,6],
    [0,3,5,4,2],
    [3,2,1,5,6]
],dtype=np.float32)

# bx=cv2.Sobel(a,cv2.CV_32F, 1, 0, ksize=3)
# by=cv2.Sobel(a,cv2.CV_32F, 0, 1, ksize=3)
# b=abs(bx)+abs(by)

b= cv2.Laplacian(a, cv2.CV_32F)

# print(bx)
# print(by)
print(b)
