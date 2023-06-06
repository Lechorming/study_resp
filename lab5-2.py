import cv2
import numpy as np

# 加载图像并转换为灰度图像
image = cv2.imread('D:\Desktop\workspace\study_resp\imgs\cameraman.tif')
# image = cv2.imread('D:\\Desktop\\workspace\\study_resp\\imgs\\room2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# gray=cv2.GaussianBlur(gray,[3,3],0)

# 定义SUSAN算法的角点检测函数
def susan_corner_detection(image, threshold=0.2):
    height, width = image.shape
    corners = np.zeros((height, width))

    # 归一化图像
    normalized_image = cv2.normalize(image.astype(np.float64), None, 0, 255, cv2.NORM_MINMAX)

    for i in range(3, height - 3):
        for j in range(3, width - 3):
            center_pixel = normalized_image[i, j]
            count = 0
            diff_sum = 0

            for u in range(-3, 4):
                for v in range(-3, 4):
                    if u == 0 and v == 0:
                        continue

                    neighbor_pixel = normalized_image[i + u, j + v]
                    difference = abs(center_pixel - neighbor_pixel)

                    if difference <= threshold:
                        count += 1
                        diff_sum += difference

            if count <= 20 and count >= 8:
                corners[i, j] = (1 - count / 24) * diff_sum

    # 非极大值抑制
    maxima = cv2.dilate(corners, None)
    corners[maxima != corners] = 0

    corners = cv2.dilate(corners, None)  # 对角点响应进行膨胀处理

    return corners

# 定义参数
threshold=70

# 应用SUSAN算法进行角点检测
corners = susan_corner_detection(gray, threshold)

cv2.imshow('original', image)

# 标记角点
image[corners > 0] = [0, 0, 255]  # 将角点标记为红色

# 显示结果
cv2.imshow('SUSAN Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
