import cv2
import numpy as np
# 读入原始图像并转换为灰度图像
gray_img = cv2.imread('D:\Desktop\workspace\study_resp\imgs\cameraman.tif',0)

# 进行FFT变换并去掉高频成分
fft_img = np.fft.fft2(gray_img)
fft_shift = np.fft.fftshift(fft_img)
magnitude_spectrum = 20 * np.log(np.abs(fft_shift))
rows, cols = gray_img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)
mask[crow - 50:crow + 50, ccol - 50:ccol + 50] = 1
fft_shift = fft_shift * mask
fft_shift = np.fft.ifftshift(fft_shift)
ifft_img = np.fft.ifft2(fft_shift)
ifft_img = np.abs(ifft_img)

# 进行DCT变换并去掉高频成分
dct_img = cv2.dct(np.float32(gray_img))
mask = np.zeros((gray_img.shape), np.float32)
mask[0:50, 0:50] = 1
dct_masked = cv2.multiply(mask,dct_img)
idct_img = cv2.idct(dct_masked)
# idct_img = np.uint8(idct_img)
import matplotlib.pyplot as plt

# 显示保留不同维度下反变换图像
plt.subplot(2, 3, 1), plt.imshow(gray_img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(np.uint8(magnitude_spectrum), cmap='gray')
plt.title('FFT Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3), plt.imshow(np.uint8(ifft_img), cmap='gray')
plt.title('IFFT Image (50x50)'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4), plt.imshow(gray_img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5), plt.imshow(dct_masked, cmap='gray')
plt.title('DCT Image (50x50)'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 6), plt.imshow(idct_img, cmap='gray')
plt.title('IDCT Image (50x50)'), plt.xticks([]), plt.yticks([])
plt.show()
