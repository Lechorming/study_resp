import cv2
import numpy as np

# 对参考图像实现均值滤波器滤波操作，需要添加相关噪声
def test1():
    # 读入图像
    img = cv2.imread("D:\Desktop\workspace\study_resp\imgs\Lenna.png", cv2.IMREAD_GRAYSCALE)

    # 添加椒盐噪声
    noisy_img = img.copy()
    noise_num = int(img.shape[0] * img.shape[1] * 0.1)  # 添加噪声的数量，此处为10%
    for i in range(noise_num):
        rand_x = np.random.randint(0, img.shape[0])
        rand_y = np.random.randint(0, img.shape[1])
        if np.random.random() < 0.5:
            noisy_img[rand_x, rand_y] = 255
        else:
            noisy_img[rand_x, rand_y] = 0

    cv2.imshow("Original", img)
    cv2.imshow("With Noise", noisy_img)
    # 对图像进行均值滤波操作
    kernel_sizes = [3,5]  # 设置滤波器大小
    for kernel_size in kernel_sizes:
        filtered_img = cv2.blur(noisy_img, (kernel_size, kernel_size))

        # 显示图像
        cv2.imshow(f"{kernel_size}*{kernel_size} Filtered", filtered_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return

# 添加高斯噪声，再使用高斯滤波器进行降噪处理
def test2():
    # 读取原始图像
    img = cv2.imread('D:\Desktop\workspace\study_resp\imgs\Lenna.png',0)

    # 添加高斯噪声
    gaussian_noise = np.random.normal(0, 10, img.shape).astype('uint8')
    img_noise = img + gaussian_noise

    # 使用高斯滤波器进行滤波
    img_filtered = cv2.GaussianBlur(img_noise, (3, 3), 0)

    # 显示原始图像、添加噪声后的图像和滤波后的图像
    cv2.imshow('Original Image', img)
    cv2.imshow('Noisy Image', img_noise)
    cv2.imshow('Filtered Image', img_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

# 使用例子图片实现中值滤波器
def test3():
    # 读入图像
    img = cv2.imread('D:\Desktop\workspace\study_resp\imgs\pepper.png', 0)

    # 添加椒盐噪声
    noisy_img = img.copy()
    noise_num = int(img.shape[0] * img.shape[1] * 0.02)  # 添加噪声的数量，此处为10%
    for i in range(noise_num):
        rand_x = np.random.randint(0, img.shape[0])
        rand_y = np.random.randint(0, img.shape[1])
        if np.random.random() < 0.5:
            noisy_img[rand_x, rand_y] = 255
        else:
            noisy_img[rand_x, rand_y] = 0

    # 中值滤波
    img_median = cv2.medianBlur(noisy_img, 3)

    # 显示原图和中值滤波后的图像
    cv2.imshow('Original Image', img)
    cv2.imshow('Noisy Image', noisy_img)
    cv2.imshow('Filtered Image', img_median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


def test4():
    # 读入灰度图像
    img = cv2.imread('D:\Desktop\workspace\study_resp\imgs\pepper.png', 0)

    # 添加高斯噪声
    gaussian_noise = np.random.normal(0, 20, size=img.shape).astype('int')
    gaussian_img = np.clip(img.astype('int') + gaussian_noise, 0, 255).astype('uint8')

    # 添加椒盐噪声
    salt_pepper_img = img.copy()
    noise_num = int(img.shape[0] * img.shape[1] * 0.1)  # 添加噪声的数量，此处为10%
    for i in range(noise_num):
        rand_x = np.random.randint(0, img.shape[0])
        rand_y = np.random.randint(0, img.shape[1])
        if np.random.random() < 0.5:
            salt_pepper_img[rand_x, rand_y] = 255
        else:
            salt_pepper_img[rand_x, rand_y] = 0

    # 添加指数分布噪声
    exponential_noise = np.random.exponential(scale=50, size=img.shape).astype('int')
    exponential_img = np.clip(img.astype('int') + exponential_noise, 0, 255).astype('uint8')

    # 添加泊松噪声
    poisson_noise = np.random.poisson(lam=50, size=img.shape).astype('int')
    poisson_img = np.clip(img.astype('int') + poisson_noise, 0, 255).astype('uint8')

    # 添加乘性噪声
    multiplicative_noise = np.random.normal(1, 0.1, size=img.shape).astype('float32')
    multiplicative_img = np.clip(img.astype('float32') * multiplicative_noise, 0, 255).astype('uint8')

    # 定义滤波器大小列表
    kernel_sizes = [5, 9, 25]

    # 定义滤波器类型列表
    filter_types = ['Mean', 'Gaussian', 'Median']

    cv2.imshow("Original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 对不同噪声和不同滤波器进行处理
    for noise_img, noise_type in zip([gaussian_img, salt_pepper_img, exponential_img, poisson_img, multiplicative_img],
                                    ['Gaussian', 'Salt and Pepper', 'Exponential', 'Poisson', 'Multiplicative']):
        cv2.imshow(f"{noise_type} Noise", noise_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for kernel_size in kernel_sizes:
            for filter_type in filter_types:
                # 构造滤波器
                if filter_type == 'Mean':
                    # kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
                    filtered_img = cv2.blur(noise_img, (kernel_size, kernel_size))
                elif filter_type == 'Gaussian':
                    # kernel = cv2.getGaussianKernel(kernel_size, 0)
                    # kernel = np.outer(kernel, kernel.transpose())
                    filtered_img=cv2.GaussianBlur(noise_img, (kernel_size, kernel_size), 0)
                elif filter_type == 'Median':
                    # kernel = np.ones((kernel_size, kernel_size), np.float32)
                    filtered_img=cv2.medianBlur(noise_img,kernel_size)

                # 应用滤波器
                # filtered_img = cv2.filter2D(noise_img, -1, kernel)
                
                # 显示结果
                cv2.imshow(f"{filter_type} Filter ({kernel_size}x{kernel_size}) with {noise_type} noise", filtered_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    return

# 尝试不使用Python自带函数，实现中值滤波器
def test5():
    def median_filter(img):
        rows, cols = img.shape
        result = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                window = img[i-1:i+2, j-1:j+2]
                result[i, j] = np.median(window)
        return result

    # 读取图像
    img = cv2.imread("D:\Desktop\workspace\study_resp\imgs\Lenna.png", 0)

    # 添加椒盐噪声
    noise_img = img.copy()
    noise_num = int(img.shape[0] * img.shape[1] * 0.1)  # 添加噪声的数量，此处为10%
    for i in range(noise_num):
        rand_x = np.random.randint(0, img.shape[0])
        rand_y = np.random.randint(0, img.shape[1])
        if np.random.random() < 0.5:
            noise_img[rand_x, rand_y] = 255
        else:
            noise_img[rand_x, rand_y] = 0

    # 中值滤波
    my_result = median_filter(noise_img)
    cv2_result= cv2.medianBlur(noise_img, 3)

    # 显示图像
    cv2.imshow("Original Image", img)
    cv2.imshow("Noise Image", noise_img)
    cv2.imshow("My Median Filter Result", my_result)
    cv2.imshow("CV2 Median Filter Result", cv2_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

test1()
test2()
test3()
test4()
test5()