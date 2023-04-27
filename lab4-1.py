import cv2
import numpy as np
from matplotlib import pyplot as plt

# 对参考图像实现均值滤波器滤波操作，需要添加相关噪声
def tests1():
    # 读取图像
    img = cv2.imread('/home/git-sync/study_resp/imgs/Lenna.png', cv2.IMREAD_GRAYSCALE)
    noise = cv2.imread('/home/git-sync/study_resp/imgs/pepper.png', cv2.IMREAD_GRAYSCALE)

    # 调整噪声图像大小和均值
    noise = cv2.resize(noise, img.shape)
    mean = np.mean(noise)

    # 添加噪声
    img_noisy = np.uint8(np.clip(np.float32(img) + np.float32(noise - mean), 0, 255))

    # 定义均值滤波器
    kernel_size = 3
    kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size**2)

    # 均值滤波
    img_filtered = cv2.filter2D(img_noisy,-1,kernel)

    # 显示结果
    plt.subplot(131),plt.imshow(img,cmap='gray'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(img_noisy,cmap='gray'),plt.title('Noisy Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img_filtered,cmap='gray'),plt.title('Filtered Image')
    plt.xticks([]), plt.yticks([])
    plt.show()
    return

# 使用高斯滤波进行降噪处理
def test2():
    # 读入图像
    img = cv2.imread('/home/git-sync/study_resp/imgs/Lenna.png')

    # 添加高斯噪声
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, img.shape)
    gaussian = gaussian.reshape(img.shape[0], img.shape[1], img.shape[2])
    img_noisy = img + gaussian

    # 高斯滤波
    img_blur = cv2.GaussianBlur(img_noisy, (5,5), 0)

    # 显示原图、添加噪声后的图像和降噪后的图像
    cv2.imshow('Original Image', img)
    cv2.imshow('Noisy Image', img_noisy.astype(np.uint8))
    cv2.imshow('Gaussian Blur', img_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

# 使用例子图片实现中值滤波器
def test3():
    # 读入图像
    img = cv2.imread('/home/git-sync/study_resp/imgs/Lenna.png', 0)

    # 中值滤波
    img_median = cv2.medianBlur(img, 3)

    # 显示原图和中值滤波后的图像
    cv2.imshow('Original Image', img)
    cv2.imshow('Median Filtered Image', img_median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test4():
    

test3()