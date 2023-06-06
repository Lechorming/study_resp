import cv2
import numpy as np
from imutils import contours

# 读取模版
templates = cv2.imread('D:\\Desktop\\workspace\\study_resp\\imgs\\templates.png')  # 读取图像
gray_tem = cv2.cvtColor(templates, cv2.COLOR_BGR2GRAY)
cv2.imshow("template", templates)

# 高斯滤波
gauss_tem=cv2.GaussianBlur(gray_tem,(5,5),1)

# 进行二值化处理
_, thresh_tem = cv2.threshold(gauss_tem, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# cv2.imshow('thresh_tem',thresh_tem)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 轮廓检测
temCnts, hierarchy = cv2.findContours(thresh_tem.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
Contours_tem=cv2.drawContours(templates, temCnts, -1, (0, 0, 255), 1)

# cv2.imshow('Contours_tem',Contours_tem)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 对轮廓进行排序
temCnts = contours.sort_contours(temCnts, method="left-to-right")[0]  # 排序，从左到右，从上到下

digits = {}
# 遍历每一个轮廓
for (i, c) in enumerate(temCnts):
    # 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    roi = thresh_tem[y:y + h, x:x + w]
    roi = cv2.resize(roi, (30, 40))
    # 每一个数字对应每一个模板
    digits[i] = roi
    # cv2.imshow('tem_roi',roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

# 读取图像
image = cv2.imread('D:\\Desktop\\workspace\\study_resp\\imgs\\digits.png')  # 读取图像
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯滤波
# gauss_img=cv2.GaussianBlur(gray_img,(1,1),1)

# 进行二值化处理
_, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

cv2.imshow("image", image)
# cv2.imshow('thresh_img',thresh_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 定义卷积核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 顶帽操作
tophat_img = cv2.morphologyEx(thresh_img, cv2.MORPH_TOPHAT, kernel)

# sobel操作
sobelx_img = cv2.Sobel(tophat_img, cv2.CV_64F, 1, 0, ksize=3)
sobelx_img = cv2.convertScaleAbs(sobelx_img)
sobely_img = cv2.Sobel(tophat_img, cv2.CV_64F, 0, 1, ksize=3)
sobely_img = cv2.convertScaleAbs(sobely_img)
sobelxy_img = cv2.addWeighted(sobelx_img, 1, sobely_img, 1, 0)

# 闭操作
closed_img = cv2.morphologyEx(sobelxy_img, cv2.MORPH_CLOSE, kernel)

# 寻找轮廓
imgCnts, hierarchy = cv2.findContours(closed_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# 将轮廓显示在原图上
Cnts_img=cv2.drawContours(image.copy(), imgCnts, -1, (0, 0, 255), 1)

def sort_contour(cnt):
    # 获取轮廓的边界框
    x, y, w, h = cv2.boundingRect(cnt)
    return (y, x)  # 返回左上角坐标 (y, x)

# 轮廓排序
# imgCnts, boundingBoxes = contours.sort_contours(imgCnts, method="top-to-bottom")
imgCnts = sorted(imgCnts, key=sort_contour)

# cv2.imshow('thresh_img',thresh_img)
# cv2.imshow('tophat_img',tophat_img)
# cv2.imshow('sobelxy_img',sobelxy_img)
# cv2.imshow('closed_img',closed_img)
# cv2.imshow('Cnts_img',Cnts_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

result_img=image.copy()
output=[]
# 对每个轮廓进行操作
for (i,ic) in enumerate(imgCnts):
    roioutput=[]
    # 计算包含轮廓的最小矩形
    (rx, ry, rw, rh) = cv2.boundingRect(ic)
    # 截取矩形所包含区域
    roi = gray_img[ry - 1:ry + rh + 1, rx - 1:rx + rw + 1]
    # 对矩形区域进行放大
    roi = cv2.resize(roi,(rw*4,rh*4))

    # 高斯滤波
    # roi=cv2.GaussianBlur(roi,(3,3),1)

    # 进行二值化处理
    _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 进行腐蚀处理
    erodekernal=np.ones((3,1),np.uint8)
    roi=cv2.erode(roi,erodekernal,iterations=1)

    # cv2.imshow('img_roi',roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # 提取区域中的每一个数字的轮廓
    digitCnts,hierarchy = cv2.findContours(roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 从左到右排序数字轮廓
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    # 识别区域中每个数字
    for dc in digitCnts:
        # 截取单个数字的矩形图像
        (dx,dy,dw,dh) = cv2.boundingRect(dc)
        roi_digit = roi[dy : dy + dh, dx:dx + dw]
        # 调整单个数字图像的大小与模版相匹配(30,40)
        roi_digit = cv2.resize(roi_digit, (30,40))
        
        # 计算单个数字图像与每个模版的匹配得分
        scores=[]
        for(digit,digitROI) in digits.items():
            result=cv2.matchTemplate(roi_digit,digitROI,cv2.TM_CCOEFF)
            (_,scr,_,_)=cv2.minMaxLoc(result)
            scores.append(scr)
        
        # 若匹配得分最大值的index是10或者匹配得分最大值低于一千万则判定该字符为'.'
        maxi=np.argmax(scores)
        if maxi==10 or np.max(scores)<1e7:
            roioutput.append('.')
        else:
            roioutput.extend(str(maxi))

        # cv2.imshow('roi_digit',roi_digit)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # 在原图中标记
    result_img=cv2.rectangle(result_img, (rx, ry),
                  (rx + rw, ry + rh), (0, 0, 255), 1)
    result_img=cv2.putText(result_img, "".join(roioutput), (rx, ry - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # 检测结果
    output.append(roioutput)

print('Image Numbers Detected: ')
k=0
for i in range(0,3):
    for j in range(0,3):
        print(''.join('{}'.format(item) for item in output[k]),end=' ')
        k+=1
    print('\n')
s=result_img.shape
result_img=cv2.resize(result_img,(s[1]*2,s[0]*2))
cv2.imshow('result_img',result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

