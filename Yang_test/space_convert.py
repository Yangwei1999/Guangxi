import cv2

from utils.plots import plot_one_box
import numpy as np

b = [[9.19000e+02, 7.31000e+02, 9.47000e+02, 7.61000e+02, 7.10782e-01, 0.00000e+00],
     [5.47000e+02, 1.05000e+03, 5.72000e+02, 1.07900e+03, 5.72106e-01, 0.00000e+00],
     [1.60200e+03, 6.69000e+02, 1.63300e+03, 7.08000e+02, 3.29563e-01, 0.00000e+00]]
img = cv2.imread(r"C:\Users\10\Desktop\yolov5-master\image1\05.png")  # \

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_color = np.array([26, 43, 46])  # 分别对应着HSV中的最小值
upper_color = np.array([34, 255, 255])  # 分别对应着HSV中的最大值
# 目的是找出黄色的部分
mask = cv2.inRange(img_hsv, lower_color, upper_color)  # inrange函数将根据最小最大HSV值检测出自己想要的颜色部分 二值化


for i in b:
    plot_one_box(i, mask, label='hat', color=[255, 0, 0],
                 line_thickness=2)

cv2.namedWindow('hsv space',0)
cv2.imshow('hsv space',mask)
cv2.waitKey(0)
cv2.imwrite('mask.png',mask)
