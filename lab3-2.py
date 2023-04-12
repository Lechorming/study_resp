import numpy as np
from matplotlib import pyplot as plt
import cv2


#(1)膨胀操作。
def test1():
    # 创建二进制图像
    BW=np.zeros((900,1000))
    BW[400:600,400:700]=1
    plt.figure("原始图像")
    plt.imshow(BW)

    # 创建膨胀核
    # 300*300的矩形膨胀核
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (300, 300))
    
    # 长度100的十字膨胀核
    k2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (100, 100))

    # 线型膨胀核
    k3 = np.zeros((150, 150), dtype=np.uint8)
    cv2.line(k3, (0, 0), (149, 149), 1, thickness=1)

    plt.figure("膨胀核1")
    plt.imshow(k1)
    plt.figure("膨胀核2")
    plt.imshow(k2)
    plt.figure("膨胀核3")
    plt.imshow(k3)

    # 膨胀操作
    new_BW1=cv2.dilate(BW,k1)
    new_BW2=cv2.dilate(BW,k2)
    new_BW3=cv2.dilate(BW,k3)
    
    plt.figure("膨胀后图像1")
    plt.imshow(new_BW1)
    plt.figure("膨胀后图像2")
    plt.imshow(new_BW2)
    plt.figure("膨胀后图像3")
    plt.imshow(new_BW3)
    plt.show()

    return
    
#（2）图像腐蚀
def test2():
    # a)对图像‘circbw.tif’(系统自带图像可以直接读取)进行腐蚀操作。
    circbw=cv2.imread("D:\Desktop\workspace\study_resp\imgs\circbw.tif")

    # 腐蚀核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 腐蚀操作
    new_circbw=cv2.erode(circbw,kernel,iterations=1)

    plt.figure("原始circbw图像")
    plt.imshow(circbw)
    plt.figure("腐蚀核")
    plt.imshow(kernel)
    plt.figure("腐蚀后图像")
    plt.imshow(new_circbw)
    plt.show()

    # b)对图像‘text.png’进行腐蚀操作。
    text=cv2.imread("D:/Desktop/workspace/study_resp/imgs/text.png",0)
    
    #腐蚀核
    line_kernel=np.zeros((5,5), dtype=np.uint8)
    cv2.line(line_kernel,(0,4),(4,0),1,thickness=1)

    # 腐蚀操作
    new_text=cv2.erode(text,line_kernel,iterations=1)
    
    plt.figure("原始text图像")
    plt.imshow(text)
    plt.figure("线型腐蚀核")
    plt.imshow(line_kernel)
    plt.figure("腐蚀后text图像")
    plt.imshow(new_text)
    plt.show()

    return

#(3)膨胀与腐蚀的综合使用
def test3():
    circbw=cv2.imread("D:\Desktop\workspace\study_resp\imgs\circbw.tif",0)
    
    # 方法一：先腐蚀(imerode)，再膨胀(imdilate)；

    # 矩形结构元素
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(20,15))
    
    # 先腐蚀后膨胀
    e_circbw=cv2.erode(circbw,kernel,iterations=1)
    d_circbw=cv2.dilate(e_circbw,kernel,iterations=1)

    cv2.imshow("circbw",circbw)
    cv2.imshow("e_circbw", e_circbw)
    cv2.imshow("d_circbw", d_circbw)
    cv2.waitKey(0)


    # 方法二：使用形态开启函数(imopen)。
    # 结构元素
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # 进行开运算
    o_circbw = cv2.morphologyEx(circbw, cv2.MORPH_OPEN, kernel)
    
    cv2.imshow("circbw",circbw)
    cv2.imshow("o_circbw", o_circbw)
    cv2.waitKey(0)


    # # c)置结构元素大小为[4 3],同时观察形态开启(imopen)与闭合(imclose)的效果，总结形态开启与闭合在图像处理中的作用。
    # 结构元素
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(4,3))

    o_circbw = cv2.morphologyEx(circbw, cv2.MORPH_OPEN, kernel)
    c_circbw = cv2.morphologyEx(circbw, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('circbw', circbw)
    cv2.imshow('o_circbw', o_circbw)
    cv2.imshow('c_circbw', c_circbw)
    cv2.waitKey(0)

    return

def bwmorph(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # 使用形态学操作进行骨架提取
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

def test4():
    # 读入图像
    img1 = cv2.imread("D:\\Desktop\\workspace\\study_resp\\imgs\\1.png",0)
    img2 = cv2.imread("D:\\Desktop\\workspace\\study_resp\\imgs\\2.png",0)
    img3 = cv2.imread("D:\\Desktop\\workspace\\study_resp\\imgs\\3.png",0)
    img4 = cv2.imread("D:\\Desktop\\workspace\\study_resp\\imgs\\4.png",0)
    img5 = cv2.imread("D:\\Desktop\\workspace\\study_resp\\imgs\\5.png",0)
    img6 = cv2.imread("D:\\Desktop\\workspace\\study_resp\\imgs\\6.png",0)

    # 提取骨架
    b1=bwmorph(img1)
    b2=bwmorph(img2)
    b3=bwmorph(img3)
    b4=bwmorph(img4)
    b5=bwmorph(img5)
    b6=bwmorph(img6)

    # 显示结果
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)
    cv2.imshow('img4', img4)
    cv2.imshow('img5', img5)
    cv2.imshow('img6', img6)
    cv2.imshow('b1', b1)
    cv2.imshow('b2', b2)
    cv2.imshow('b3', b3)
    cv2.imshow('b4', b4)
    cv2.imshow('b5', b5)
    cv2.imshow('b6', b6)
    cv2.waitKey(0)
    return

test1()
test2()
test3()
test4()
