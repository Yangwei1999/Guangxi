# 找出框 对框进行黄色二值化  根据阈值判断误判  以10.png为例子
import cv2
import numpy as np

# img_test = img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
# 我可以认为超过5% 就是实际的安全帽
a = [[9.51000e+02, 6.92000e+02, 9.73000e+02, 7.21000e+02, 5.96407e-01, 0.00000e+00],
     [5.46000e+02, 1.04900e+03, 5.72000e+02, 1.07800e+03, 4.10367e-01, 0.00000e+00],
     [1.60100e+03, 6.71000e+02, 1.63300e+03, 7.10000e+02, 3.28861e-01, 0.00000e+00]]
b = [9.51000e+02, 6.92000e+02, 9.73000e+02, 7.21000e+02, 5.96407e-01, 0.00000e+00]


def coutning_num(img):
    count = 0
    n, m = img.shape
    for i in img:
        for j in i:
            if (j == 255):
                count = count + 1
    count = count / (n * m)
    return count


def double_check_HSV(img, xyxys):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([26, 43, 46])  # 分别对应着HSV中的最小值
    upper_color = np.array([34, 255, 255])  # 分别对应着HSV中的最大值
    # 目的是找出黄色的部分
    mask = cv2.inRange(img_hsv, lower_color, upper_color)  # inrange函数将根据最小最大HSV值检测出自己想要的颜色部分 二值化
    # 争对已经找出的这些框 把这些框 分别计算其中的比例  比例小的 直接去掉
    img_part = mask[int(xyxys[1]):int(xyxys[3]), int(xyxys[0]):int(xyxys[2])]

    ratio_color = coutning_num(img_part)
    print('ratio color :',ratio_color)
    return ratio_color


b =[1.29000e+02, 7.40000e+01, 1.65000e+02, 1.17000e+02, 2.92030e-01, 0.00000e+00]
img = cv2.imread(r"C:\Users\10\Desktop\yolov5-master\image1\05.png")  #
double_check_HSV(img, b)
