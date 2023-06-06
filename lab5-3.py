import cv2
import numpy as np

# 加载图像并转换为灰度图像
# image = cv2.imread('D:\Desktop\workspace\study_resp\imgs\cameraman.tif')
image = cv2.imread('D:\\Desktop\\workspace\\study_resp\imgs\\room2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 定义Moravec算法的角点检测函数
def moravec_corner_detection(image, window_size=3, threshold=1000):
    height, width = image.shape
    corners = np.zeros((height, width))

    for i in range(window_size // 2, height - window_size // 2):
        for j in range(window_size // 2, width - window_size // 2):
            min_diff = np.int64(1e9)
            # min_diff = np.int64(0)

            diff=np.zeros((4))
            for k in range(1,window_size//2+1):
                # diff[0]+=(image[i, j-k] - image[i, j-k+1]) ** 2
                # diff[1]+=(image[i, j+k] - image[i, j+k-1]) ** 2
                # diff[2]+=(image[i-k, j] - image[i-k+1, j]) ** 2
                # diff[3]+=(image[i+k, j] - image[i+k-1, j]) ** 2
                diff[0]+=(image[i, j-k] - image[i, j]) ** 2
                diff[1]+=(image[i, j+k] - image[i, j]) ** 2
                diff[2]+=(image[i-k, j] - image[i, j]) ** 2
                diff[3]+=(image[i+k, j] - image[i, j]) ** 2

            for k in range(4):
                if min_diff > diff[k]:
                    min_diff = diff[k]

            if min_diff > threshold:
                corners[i, j] = 255

    # 对角点响应进行膨胀处理
    corners = cv2.dilate(corners, None) 

    return corners

# 定义参数
window_size=3
threshold=2000

# 应用Moravec算法进行角点检测 
corners = moravec_corner_detection(gray.astype(np.int64), window_size, threshold)

cv2.imshow('original', image)

# 标记角点
image[corners == 255] = [0, 0, 255]  # 将角点标记为红色

# 显示结果
cv2.imshow('Moravec Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
