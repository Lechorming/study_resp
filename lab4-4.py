import cv2
import numpy as np
import matplotlib.pyplot as plt


def test1():
    # 读取图像
    img = cv2.imread("D:/Desktop/workspace/study_resp/imgs/room.png", 0)

    # Roberts算子进行边缘检测
    roberts_kernel_x = np.array([[1, 0], [0, -1]], np.float32)
    roberts_kernel_y = np.array([[0, 1], [-1, 0]], np.float32)
    img_roberts_x = np.float32(cv2.filter2D(img, -1, roberts_kernel_x))
    img_roberts_y = np.float32(cv2.filter2D(img, -1, roberts_kernel_y))

    # 求梯度模，使用欧几里德距离方式计算
    euclideandis = np.float32(np.sqrt(img_roberts_x**2 + img_roberts_y**2))

    # 使用街区距离方式计算
    blockdis = np.float32(
        cv2.addWeighted(
            cv2.convertScaleAbs(img_roberts_x),
            0.5,
            cv2.convertScaleAbs(img_roberts_y),
            0.5,
            0,
        )
    )

    # 将梯度模归一化到[0, 255]范围内，并转换为整型
    euclideandis = np.uint8(cv2.normalize(
        euclideandis, None, 0, 255, cv2.NORM_MINMAX))
    blockdis = np.uint8(cv2.normalize(blockdis, None, 0, 255, cv2.NORM_MINMAX))

    # 显示水平和垂直检测结果的直方图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(img_roberts_x.ravel(), bins=256, range=(0, 255))
    plt.title("Horizontal Edge Detection Result")
    plt.subplot(1, 2, 2)
    plt.hist(img_roberts_y.ravel(), bins=256, range=(0, 255))
    plt.title("Vertical Edge Detection Result")
    plt.show()

    # 显示欧几里得距离和街区距离的直方图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(euclideandis.ravel(), bins=256, range=(0, 255))
    plt.title("Euclidean Distance")
    plt.subplot(1, 2, 2)
    plt.hist(blockdis.ravel(), bins=256, range=(0, 255))
    plt.title("Manhattan Distance")
    plt.show()

    # 二值化欧几里得距离和街区距离的计算结果
    _, euclideandis = cv2.threshold(euclideandis, 30, 255, cv2.THRESH_BINARY)
    _, blockdis = cv2.threshold(blockdis, 30, 255, cv2.THRESH_BINARY)

    # 将像素值翻转
    euclideandis = 255 - euclideandis
    blockdis = 255 - blockdis

    # 显示原始图像和Roberts算子处理后的图像
    cv2.imshow("Original Image", img)
    cv2.imshow("Roberts x", np.uint8(img_roberts_x))
    cv2.imshow("Roberts y", np.uint8(img_roberts_y))

    cv2.imshow("Roberts Edge Detection with Euclidean Distadnce", euclideandis)
    cv2.imshow("Roberts Edge Detection with Block Distance", blockdis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


test1()
