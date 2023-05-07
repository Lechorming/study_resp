import numpy as np
import matplotlib.pyplot as plt
import cv2

def test1():
    f = np.zeros((30,30))
    f[4:24,12:17] = 1
    plt.imshow(f, cmap='gray', aspect='equal')
    F = np.fft.fft2(f)
    F2 = np.log(np.abs(F))
    plt.figure()
    plt.imshow(F2, cmap='jet', vmin=-1, vmax=5, aspect='equal')

    F = np.fft.fft2(f, (256, 256))
    plt.figure()
    plt.imshow(np.log(np.abs(F)), cmap='jet', vmin=-1, vmax=5, aspect='equal')

    F2 = np.fft.fftshift(F)
    plt.figure()
    plt.imshow(np.log(np.abs(F2)), cmap='jet', vmin=-1, vmax=5, aspect='equal')
    plt.show()
    return

def test2():
    bw = cv2.imread('D:/Desktop/workspace/study_resp/imgs/text.png', 0)

    a = bw[32:45, 87:99]

    plt.imshow(bw, cmap='gray')
    plt.figure()
    plt.imshow(a, cmap='gray')

    fft_bw = np.fft.fft2(bw)
    fft_a = np.fft.fft2(np.rot90(a, 2), (256, 256))

    C = np.real(np.fft.ifft2(fft_bw * fft_a))  # 求相关性

    plt.figure()
    plt.imshow(C, cmap='gray')

    thresh = np.max(C)

    plt.figure()
    plt.imshow(C > thresh - 10, cmap='gray')
    plt.figure()
    plt.imshow(C > thresh - 1000000, cmap='gray')

    plt.show()

    return

def test3():
    # 读取图像
    RGB = cv2.imread('D:/Desktop/workspace/study_resp/imgs/autumn.tif')

    # 显示原始图像
    plt.imshow(cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB))
    plt.show()

    # 转换为灰度图像
    I = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)

    # 显示灰度图像
    plt.imshow(I, cmap='gray')
    plt.show()

    # 进行离散余弦变换
    J = cv2.dct(np.float32(I))

    # 显示离散余弦变换结果的幅度谱
    plt.imshow(np.log(abs(J)), cmap='jet', vmin=-2, vmax=8)
    plt.colorbar()
    plt.show()

    return

def test4():
    # 读取图像
    RGB = cv2.imread('D:/Desktop/workspace/study_resp/imgs/autumn.tif')

    # 转换为灰度图像
    I = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)

    # 进行DCT变换
    J = cv2.dct(np.float32(I))

    # 显示原图像
    plt.imshow(I, cmap='gray')
    plt.title('Original Image')
    plt.show()

    # 进行IDCT逆变换
    K = cv2.idct(J)

    # 显示IDCT逆变换后的图像
    plt.imshow(K, cmap='gray', vmin=0, vmax=255)
    plt.title('IDCT Image')
    plt.show()

    # 舍弃系数
    J[np.abs(J) < 10] = 0

    # 进行IDCT逆变换
    K2 = cv2.idct(J)

    # 显示舍弃系数后的IDCT逆变换后的图像
    plt.imshow(K2, cmap='gray', vmin=0, vmax=255)
    plt.title('IDCT Image (with coefficients removed)')
    plt.show()

    return

def test5():
    # 读取图像
    img = cv2.imread('D:\Desktop\workspace\study_resp\imgs\cameraman.tif', cv2.IMREAD_GRAYSCALE)

    # 将图像转换为浮点数类型
    img = np.float32(img)/255.0

    # 对图像进行 DCT 变换
    dct = img.copy()
    for i in range(0, img.shape[0], 8):
        for j in range(0, img.shape[1], 8):
            dct[i:i+8, j:j+8] = cv2.dct(img[i:i+8,j:j+8])
    # 构造掩膜矩阵
    mask = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

    # 对变换系数进行掩膜操作
    dct_masked = dct.copy()
    for i in range(0, dct.shape[0], 8):
        for j in range(0, dct.shape[1], 8):
            dct_masked[i:i+8, j:j+8] = cv2.multiply(mask,dct[i:i+8,j:j+8])

    # 对掩膜操作后的变换系数进行反DCT变换
    img_dct_filtered = dct_masked.copy()
    for i in range(0, dct_masked.shape[0], 8):
        for j in range(0, dct_masked.shape[1], 8):
            img_dct_filtered[i:i+8, j:j+8] = cv2.idct(dct_masked[i:i+8, j:j+8])

    # 显示压缩前后的图像
    cv2.imshow('original', img)
    cv2.imshow('DCT filtered', img_dct_filtered)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return

def test6():
    # 读取图像
    img = cv2.imread('D:\Desktop\workspace\study_resp\imgs\cameraman.tif', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original', img)
    for k in range(1,4):
        # 进行FFT变换并去掉高频成分
        fft_img = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft_img)
        magnitude_spectrum = 20 * np.log(np.abs(fft_shift))
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - k*25:crow + k*25, ccol - k*25:ccol + k*25] = 1
        fft_shift = fft_shift * mask
        fft_shift = np.fft.ifftshift(fft_shift)
        ifft_img = np.fft.ifft2(fft_shift)
        ifft_img = np.abs(ifft_img)

        # 对图像进行 DCT 变换并去掉高频成分
        dct=cv2.dct(np.float32(img)/255.0)
        mask=np.zeros(img.shape,dtype=np.float32)
        mask[0:k*50,0:k*50]=1
        dct_masked=cv2.multiply(mask,dct)
        img_dct_filtered=cv2.idct(dct_masked)

        # 显示压缩前后的图像
        cv2.imshow('FFT Magnitude Spectrum',np.uint8(magnitude_spectrum))
        cv2.imshow(f'IFFT in {k*50}*{k*50}',np.uint8(ifft_img))
        cv2.imshow('DCT',dct)
        cv2.imshow(f'IDCT in {k*50}*{k*50}', img_dct_filtered)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return

test1()
test2()
test3()
test4()
test5()
test6()