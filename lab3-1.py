import cv2
import numpy as np
import matplotlib.pyplot as plt

#1.直方图显示，显示灰度级n=64,n=128,n=256的图像直方图
def test1():
    img=cv2.imread('D:\Desktop\workspace\study_resp\imgs\Lenna.jpg')

    cv2.imshow('img',img)

    hist256=cv2.calcHist([img],[0],None,[256],[0,256])
    hist128=cv2.calcHist([img],[0],None,[128],[0,256])
    hist64=cv2.calcHist([img],[0],None,[64],[0,256])

    plt.figure("灰度值n=256直方图")
    plt.plot(hist256)
    plt.figure("灰度值n=128直方图")
    plt.plot(hist128)
    plt.figure("灰度值n=64直方图")
    plt.plot(hist64)
    plt.show()

#2.利用函数imadjust调解图像灰度范围，观察变换后的图像及其直方图的变化
def my_imadjust(img,x,y,a,b,gamma,c):
    if(not(x<y) or not(a<b)):
        return img
    w, h = img.shape
    img1 = img.copy()
    # imadjust函数运算部分
    for i in range(0, w):
        for j in range(0, h):
            if img[i, j] <= x:
                img1[i, j] = a
            elif img[i, j] >= y:
                img1[i, j] = b
            else:
                img1[i, j] = c * (img[i, j]**gamma)
    return img1
    
def test2():
    img=cv2.imread('D:\Desktop\workspace\study_resp\imgs\Lenna.jpg',0)
    result =my_imadjust(img,10,210,0,255,1,1)
    hist=cv2.calcHist([result],[0],None,[256],[0,255])
    histo=cv2.calcHist([img],[0],None,[256],[0,255])

    cv2.imshow("origin",img)
    cv2.imshow("result",result)
    cv2.waitKey(0)

    plt.figure("原始图像直方图")
    plt.plot(histo)

    plt.figure("imadjust变换后图像")
    plt.plot(hist)
    plt.show()

#3.分别对图像‘flower.tif’和‘Lenna.jpg’进行直方图均衡化处理，
# 比较处理前后图像及直方图分布的变化。
def test3():
    img1=cv2.imread("D:/Desktop/workspace/study_resp/imgs/flower.tif",0)
    img1_equ=cv2.equalizeHist(img1)
    img2=cv2.imread("D:/Desktop/workspace/study_resp/imgs/Lenna.jpg",0)
    img2_equ=cv2.equalizeHist(img2)
    # hist1=cv2.calcHist([img1],[0],None,[256],[0,255])
    # hist1_equ=cv2.calcHist([img1_equ],[0],None,[256],[0,255])
    # hist2=cv2.calcHist([img2],[0],None,[256],[0,255])
    # hist2_equ=cv2.calcHist([img2_equ],[0],None,[256],[0,255])

    cv2.imshow("origin flower",img1)
    cv2.imshow("processed flower",img1_equ)
    cv2.waitKey(0)
    plt.figure("flower.tif原始直方图")
    plt.hist(img1.ravel(),256)
    # plt.plot(hist1)
    plt.figure("flower.tif均衡化后直方图")
    plt.hist(img1_equ.ravel(),256)
    # plt.plot(hist1_equ)
    plt.show()
    cv2.imshow("origin lenna",img2)
    cv2.imshow("processed lenna",img2_equ)
    cv2.waitKey(0)
    plt.figure("lenna.jpg原始直方图")
    plt.hist(img2.ravel(),256)
    # plt.plot(hist2)
    plt.figure("lenna.jpg均衡化后直方图")
    plt.hist(img2_equ.ravel(),256)
    # plt.plot(hist2_equ)
    plt.show()

#4.读取一幅彩色图像，对RGB图像的每个通道进行直方图均衡化,对均衡化后进行
# 重新合并成彩色图像，展示不同阶段的图像效果。
# 另将RGB图像转换为HSV图像（rgb2hsv函数），分别对三分量的图像行直方图
# 均衡化，最后合并成新的彩色图像，分析不同阶段的图像效果。
def test4():
    img =cv2.imread("D:/Desktop/workspace/study_resp/imgs/flower.tif")
    # 转换成直方图
    histb=cv2.calcHist([img],[0],None,[256],[0,255])
    histg=cv2.calcHist([img],[1],None,[256],[0,255])
    histr=cv2.calcHist([img],[2],None,[256],[0,255])
    
    #1、拆分通道
    imgb,imgg,imgr=cv2.split(img)
    
    #2、均衡化处理
    imgb_equ=cv2.equalizeHist(imgb)
    imgg_equ=cv2.equalizeHist(imgg)
    imgr_equ=cv2.equalizeHist(imgr)
    
    #3、合并
    img_equ=cv2.merge([imgb_equ,imgg_equ,imgr_equ])
    # 转换成直方图
    histb_equ=cv2.calcHist([img_equ],[0],None,[256],[0,255])
    histg_equ=cv2.calcHist([img_equ],[1],None,[256],[0,255])
    histr_equ=cv2.calcHist([img_equ],[2],None,[256],[0,255])
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_equ_rgb = cv2.cvtColor(img_equ, cv2.COLOR_BGR2RGB)
    
    plt.figure("original图片")
    plt.imshow(img_rgb)

    plt.figure("original直方图")
    plt.plot(histb,color="b")
    plt.plot(histg,color="g")
    plt.plot(histr,color="r")

    # plt.figure("blue图片")
    # plt.imshow(imgb)
    # plt.figure("green图片")
    # plt.imshow(imgg)
    # plt.figure("red图片")
    # plt.imshow(imgr)
    # cv2.imshow("imgb",imgb)
    # cv2.imshow("imgg",imgg)
    # cv2.imshow("imgr",imgr)
    # cv2.waitKey(0)

    plt.figure("processed图片")
    plt.imshow(img_equ_rgb)

    plt.figure("processed直方图")
    plt.plot(histb_equ,color="b")
    plt.plot(histg_equ,color="g")
    plt.plot(histr_equ,color="r")

    # plt.figure("blue均衡化后图片")
    # plt.imshow(imgb_equ)
    # plt.figure("green均衡化后图片")
    # plt.imshow(imgg_equ)
    # plt.figure("red均衡化后图片")
    # plt.imshow(imgr_equ)
    # plt.show()

    # hsv图
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    histh_hsv=cv2.calcHist([img_hsv],[0],None,[180],[0,179])
    hists_hsv=cv2.calcHist([img_hsv],[1],None,[256],[0,255])
    histv_hsv=cv2.calcHist([img_hsv],[2],None,[256],[0,255])

    #直方图均衡
    imgh,imgs,imgv=cv2.split(img_hsv)
    # imgh_equ=cv2.equalizeHist(imgh)
    imgh_equ=imgh
    imgs_equ=cv2.equalizeHist(imgs)
    imgv_equ=cv2.equalizeHist(imgv)

    #合并#
    img_hsv_equ=cv2.merge([imgh_equ,imgs_equ,imgv_equ])
    histh_hsv_equ=cv2.calcHist([img_hsv_equ],[0],None,[180],[0,179])
    hists_hsv_equ=cv2.calcHist([img_hsv_equ],[1],None,[256],[0,255])
    histv_hsv_equ=cv2.calcHist([img_hsv_equ],[2],None,[256],[0,255])

    img_gbr2hsv=cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    img_gbr2hsv_equ=cv2.cvtColor(img_hsv_equ, cv2.COLOR_HSV2RGB)

    plt.figure("original图片")
    plt.imshow(img_gbr2hsv)
    plt.figure("original直方图")
    plt.plot(histh_hsv,color="b")
    plt.plot(hists_hsv,color="g")
    plt.plot(histv_hsv,color="r")

    plt.figure("processed图片")
    plt.imshow(img_gbr2hsv_equ)
    plt.figure("processed直方图")
    plt.plot(histh_hsv_equ,color="b")
    plt.plot(hists_hsv_equ,color="g")
    plt.plot(histv_hsv_equ,color="r")

    plt.show()

# (5)自行设计程序实现图像的直方图均衡（选做）。

# 对灰度图像进行均衡的操作
def hist_equal(img, z_max=255):
	H, W = img.shape
	S = H * W  * 1.
	out = img.copy()
	sum_h = 0.
	for i in range(1, 255):
		ind = np.where(img == i)
		sum_h += len(img[ind])
		z_prime = z_max / S * sum_h
		out[ind] = z_prime
	out = out.astype(np.uint8)
	return out

def test5():
    img =cv2.imread("D:/Desktop/workspace/study_resp/imgs/elephant.jpg")
    imgb,imgg,imgr=cv2.split(img)
    
    histb = cv2.calcHist([img], [0], None, [256], [0, 255])
    histg = cv2.calcHist([img], [1], None, [256], [0, 255])
    histr = cv2.calcHist([img], [2], None, [256], [0, 255])

    new_imgb=hist_equal(imgb)
    new_imgg=hist_equal(imgg)
    new_imgr=hist_equal(imgr)
    new_img=cv2.merge([new_imgb,new_imgg,new_imgr])

    new_histb = cv2.calcHist([new_img], [0], None, [256], [0, 255])
    new_histg = cv2.calcHist([new_img], [1], None, [256], [0, 255])
    new_histr = cv2.calcHist([new_img], [2], None, [256], [0, 255])

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_img_rgb = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

    plt.figure("原始图像")
    plt.imshow(img_rgb)
    plt.figure("均衡化后图像")
    plt.imshow(new_img_rgb)

    plt.figure("原始直方图")
    plt.plot(histb,color="blue")
    plt.plot(histg,color="green")
    plt.plot(histr,color="red")

    plt.figure("自行设计的直方图均衡化")
    plt.plot(new_histb,color="blue")
    plt.plot(new_histg,color="green")
    plt.plot(new_histr,color="red")
    plt.show()

# test1()
# test2()
# test3()
# test4()
test5()

