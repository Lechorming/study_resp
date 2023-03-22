import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('Lenna.jpg')

cv2.imshow('img1',img)

hist=cv2.calcHist([img],[0],None,[256],[0,256])

plt.figure(figsize=(10,8))
plt.plot(hist)
plt.show()