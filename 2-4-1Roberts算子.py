import cv2
import numpy as np
import matplotlib.pyplot  as plt
# 加载图像
img = cv2.imread('D:/Desktop/workspace/study_resp/imgs/room.png', cv2.IMREAD_GRAYSCALE)

# 加入零均值的高斯噪声
noise = np.zeros_like(img)
cv2.randn(noise, 0, 10)
noisy_img = cv2.add(img, noise)

a = int(input("不加噪输入0，加噪输入1\n"))
if a == 0:
    ori_img = img
elif a == 1:
    ori_img = noisy_img
    cv2.imshow("noisy_img", noisy_img)

# Roberts算子边缘检测
roberts_x = np.array([[0, 1], [-1, 0]])
roberts_y = np.array([[1, 0], [0, -1]])
edge_x = cv2.filter2D(ori_img, -1, roberts_x)
edge_y = cv2.filter2D(ori_img, -1, roberts_y)

# 将处理结果转化为“白底黑线条”的方式
edge_x1 = 255 - edge_x
edge_y1 = 255 - edge_y

# 显示处理后的水平边界和垂直边界检测结果
cv2.imshow("Horizontal Edges", edge_x1)
cv2.imshow("Vertical Edges", edge_y1)

# 求梯度模，使用欧几里德距离方式计算
# edge_x = edge_x / np.max(edge_x)
# edge_y = edge_y / np.max(edge_y)
grad_mod_euclidean = np.float32(np.sqrt(edge_x**2 + edge_y**2))

# 使用街区距离方式计算
grad_mod_manhattan = np.float32(cv2.addWeighted(cv2.convertScaleAbs(edge_x), 0.5, cv2.convertScaleAbs(edge_y), 0.5, 0))

# 显示梯度模# grad_mod_euclidean = np.sqrt(grad_x**2 + grad_y**2)
cv2.imshow("Euclidean Gradient Modulus", grad_mod_euclidean)
cv2.imshow("Manhattan Gradient Modulus", grad_mod_manhattan)

# 对梯度模进行二值化处理，并显示处理结果  #了解二值图像中的参数
if a==1:
    _, binary_img_gme = cv2.threshold(grad_mod_euclidean, 12, 255, cv2.THRESH_BINARY)
    _, binary_img_gmm = cv2.threshold(grad_mod_manhattan, 18, 255, cv2.THRESH_BINARY)
elif a==0:
    _, binary_img_gme = cv2.threshold(grad_mod_euclidean, 10, 255, cv2.THRESH_BINARY)
    _, binary_img_gmm = cv2.threshold(grad_mod_manhattan, 11, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary Euclidean Gradient0 Modulus", binary_img_gme)
cv2.imshow("Binary Manhattan Gradient Modulus", binary_img_gmm)

# 绘制灰度直方图
plt.subplot(1, 2, 1)
plt.hist(grad_mod_euclidean.ravel(), bins=256, range=(0, 255))
plt.title('Euclidean Distance')
plt.subplot(1, 2, 2)
plt.hist(grad_mod_manhattan.ravel(), bins=256, range=(0, 255))
plt.title('Manhattan Distance')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
