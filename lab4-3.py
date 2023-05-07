import cv2
import numpy as np

# 读入灰度图像
img1 = cv2.imread('D:\Desktop\workspace\study_resp\imgs\cameraman.tif', 0)
img2 = cv2.imread('D:\Desktop\workspace\study_resp\imgs\Lenna.png', 0)

# 添加高斯噪声
def add_gaussiannoise(img,mean=0,sigma=20):
    gaussian_noise = np.random.normal(mean, sigma, size=img.shape).astype('int')
    gaussian_img = np.clip(img.astype('int') + gaussian_noise, 0, 255).astype('uint8')
    return gaussian_img

# 添加椒盐噪声
def add_saltpeppernoise(img,sigma=0.1):# 添加噪声的数量，此处为10%
    salt_pepper_img = img.copy()
    noise_num = int(img.shape[0] * img.shape[1] * sigma)  
    for i in range(noise_num):
        rand_x = np.random.randint(0, img.shape[0])
        rand_y = np.random.randint(0, img.shape[1])
        if np.random.random() < 0.5:
            salt_pepper_img[rand_x, rand_y] = 255
        else:
            salt_pepper_img[rand_x, rand_y] = 0
    return salt_pepper_img

# 添加指数分布噪声
def add_exponentialnoise(img):
    exponential_noise = np.random.exponential(scale=50, size=img.shape).astype('int')
    exponential_img = np.clip(img.astype('int') + exponential_noise, 0, 255).astype('uint8')
    return exponential_img

# butterworth滤波器
def butterworth_filter(img, filter_type='low', D0=10, n=1):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    D = np.sqrt(x**2 + y**2)
    if filter_type == 'low':
        H = 1 / (1 + (D/D0)**(2*n))
    elif filter_type == 'high':
        H = 1 / (1 + (D0/D)**(2*n))
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    Gshift = H * Fshift
    G = np.fft.ifftshift(Gshift)
    g = np.fft.ifft2(G)
    g = np.real(g)
    g = np.uint8(np.clip(g, 0, 255))
    return g


noisetypes=['gaussian','saltpepper','exponential']

for img in [img1,img2]:
    cv2.imshow('original',img)
    for noisetype in noisetypes:
        if noisetype=='gaussian':
            noise_img=add_gaussiannoise(img)
        elif noisetype=='saltpepper':
            noise_img=add_saltpeppernoise(img)
        elif noisetype=='exponential':
            noise_img=add_exponentialnoise(img)

        lowpass_img = butterworth_filter(noise_img, filter_type='low', D0=50, n=2)
        highpass_img = butterworth_filter(noise_img, filter_type='high', D0=5, n=2)

        cv2.imshow(f'{noisetype} noise img',noise_img)
        cv2.imshow('lowpass',lowpass_img)
        cv2.imshow('highpass',highpass_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
