import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('D:\Desktop\workspace\study_resp\Lenna.jpg')

cv2.imshow('img1',img)

hist256=cv2.calcHist([img],[0],None,[256],[0,256])
hist128=cv2.calcHist([img],[0],None,[128],[0,256])
hist64=cv2.calcHist([img],[0],None,[64],[0,256])

plt.figure(1,figsize=(10,8))
plt.plot(hist256)
plt.figure(2,figsize=(10,8))
plt.plot(hist128)
plt.figure(3,figsize=(10,8))
plt.plot(hist64)
plt.show()