import cv2
import numpy as np

# 加载图像并转换为灰度图像
image = cv2.imread('D:\Desktop\workspace\study_resp\imgs\cameraman.tif')
# image = cv2.imread('D:\\Desktop\\workspace\\study_resp\\imgs\\room2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算图像梯度
dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# 计算Harris矩阵的三个分量
dx2 = dx * dx
dy2 = dy * dy
dxy = dx * dy

# 定义计算角点响应函数的参数
k = 0.04  # Harris角点响应函数参数
window_size = 3  # 邻域窗口大小

# 计算角点响应函数
corner_response = np.zeros(gray.shape)
offset = window_size // 2
for i in range(offset, gray.shape[0] - offset):
    for j in range(offset, gray.shape[1] - offset):
        sum_dx2 = np.sum(dx2[i - offset:i + offset + 1, j - offset:j + offset + 1])
        sum_dy2 = np.sum(dy2[i - offset:i + offset + 1, j - offset:j + offset + 1])
        sum_dxy = np.sum(dxy[i - offset:i + offset + 1, j - offset:j + offset + 1])

        det = sum_dx2 * sum_dy2 - sum_dxy ** 2
        trace = sum_dx2 + sum_dy2
        corner_response[i, j] = det - k * (trace ** 2)

cv2.imshow('original', image)

# 设定阈值并标记角点
threshold = 0.01 * corner_response.max()
image[corner_response > threshold] = [0, 0, 255]  # 将角点标记为红色

# 显示结果
cv2.imshow('Harris Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
