import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack


# 二维Butterworth低通滤波器
def butterworth_lp_filter(size, cutoff, n):
    """
    :param size: 滤波器的大小
    :param cutoff: 截止频率
    :param n: 滤波器的阶数
    """
    rows, cols = size
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    radius = np.sqrt(x ** 2 + y ** 2)
    lpf = 1 / (1 + (radius / cutoff) ** (2 * n))
    return lpf

# 二维Butterworth高通滤波器
def butterworth_hp_filter(size, cutoff, n):
    rows, cols = size
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    radius = np.sqrt(x ** 2 + y ** 2)
    hpf = 1 / (1 + (cutoff / radius) ** (2 * n))
    return hpf

# 读取灰度图像
img = cv2.imread("D:\Desktop\workspace\study_resp\imgs\Lenna.png", cv2.IMREAD_GRAYSCALE)

# 定义不同类型的噪声
noise_types = ["gaussian", "salt_and_pepper", "exponential"]

# 添加不同类型的噪声与滤波
for noise_type in noise_types:
    # 添加噪声
    if noise_type == "gaussian":
        noise = np.zeros_like(img)
        cv2.randn(noise, 0, 50)
        img_noisy = cv2.add(img, noise)
    elif noise_type == "salt_and_pepper":
        for i in range(500):
            # 随机取椒盐噪声的位置
            rand_x = np.random.randint(0, img.shape[0])
            rand_y = np.random.randint(0, img.shape[1])
            if np.random.randint(0, 2) == 0:
                img_noisy[rand_x, rand_y] = 0
            else:
                img_noisy[rand_x, rand_y] = 255
    elif noise_type == "exponential":
        scale = 50
        noise = np.random.exponential(scale, img.shape)
        img_noisy = cv2.add(img, noise.astype(np.uint8))

    # 对添加噪声的图像进行傅里叶变换
    f = fftpack.fft2(img_noisy)

    # 应用高低滤波器
    for filter_type in ["H", "L"]:
        if filter_type == "H":
            # 将高通滤波器应用于傅里叶变换后的图像
            hpf = butterworth_hp_filter(img_noisy.shape, cutoff=50, n=2)
            hpf_shift = fftpack.ifftshift(hpf)
            f_hpf_filtered = f * hpf_shift
            img_filtered = np.real(fftpack.ifft2(f_hpf_filtered))
        elif filter_type == "L":
            # 将低通滤波器应用于傅里叶变换后的图像
            lpf = butterworth_lp_filter(img_noisy.shape, cutoff=50, n=2)
            lpf_shift = fftpack.ifftshift(lpf)
            f_lpf_filtered = f * lpf_shift
            img_filtered = np.real(fftpack.ifft2(f_lpf_filtered))

        # 显示图像
        plt.figure(figsize=(6, 6)), plt.imshow(img_noisy, cmap='gray'), plt.title(f"{noise_type}")
        plt.axis('off')
        plt.figure(figsize=(6, 6)), plt.imshow(img_filtered, cmap='gray'), plt.title(f"{filter_type} filter and {noise_type} noise")
        plt.axis('off')
        plt.show()
