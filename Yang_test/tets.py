import os
import cv2

# names ['hat', 'person']
# [640, 640]
# tensor([[1.63100e+03, 5.27000e+02, 1.65600e+03, 5.55000e+02, 7.37299e-01, 0.00000e+00],
#         [6.18000e+02, 5.08000e+02, 6.43000e+02, 5.37000e+02, 6.05086e-01, 0.00000e+00],
#         [1.29000e+02, 7.40000e+01, 1.65000e+02, 1.17000e+02, 2.92030e-01, 0.00000e+00]], device='cuda:0')
import matplotlib.pyplot as plt
import matplotlib

from utils.plots import plot_one_box

matplotlib.use('TkAgg')
img = cv2.imread(r'C:\Users\10\Desktop\yolov5-master\image1\01.png')
img_plot = img.copy()


a = [[1.63100e+03, 5.27000e+02, 1.65600e+03, 5.55000e+02, 7.37299e-01, 0.00000e+00],
     [6.18000e+02, 5.08000e+02, 6.43000e+02, 5.37000e+02, 6.05086e-01, 0.00000e+00],
     [1.29000e+02, 7.40000e+01, 1.65000e+02, 1.17000e+02, 2.92030e-01, 0.00000e+00]]

color = ('b', 'g', 'r')
k = 0

for xyxy in a:
    k = k + 1
    plot_one_box(xyxy, img_plot, label='test', color=[255, 0, 0],
                 line_thickness=2)

    img_test = img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
    for i, col in enumerate(color):
        histr = cv2.calcHist([img_test], [i], None, [256], [0, 256]) / (
                    (int(xyxy[3]) - int(xyxy[1]) + 1) * (int(xyxy[2]) - int(xyxy[0]) + 1))
        plt.subplot(2, 2, k), plt.plot(histr, color=col)
        plt.xlim([0, 256]), plt.title('Histogram')

b = [[9.19000e+02, 7.31000e+02, 9.47000e+02, 7.61000e+02, 7.10782e-01, 0.00000e+00],
     [5.47000e+02, 1.05000e+03, 5.72000e+02, 1.07900e+03, 5.72106e-01, 0.00000e+00],
     [1.60200e+03, 6.69000e+02, 1.63300e+03, 7.08000e+02, 3.29563e-01, 0.00000e+00]]
k = 0

img = cv2.imread(r'C:\Users\10\Desktop\yolov5-master\image1\05.png')
img_plot = img.copy()
plt.figure()
for xyxy in b:
    k = k + 1
    plot_one_box(xyxy, img_plot, label='test', color=[255, 0, 0],
                 line_thickness=2)

    img_test = img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
    for i, col in enumerate(color):
        histr = cv2.calcHist([img_test], [i], None, [256], [0, 256]) / (
                    (int(xyxy[3]) - int(xyxy[1]) + 1) * (int(xyxy[2]) - int(xyxy[0]) + 1))
        plt.subplot(2, 2, k), plt.plot(histr, color=col)
        plt.xlim([0, 256]), plt.title('Histogram')
plt.show()

# cv2.imshow('sad', img_test)
# cv2.waitKey(0)

# cv2.imshow('sad',img_plot)
# cv2.waitKey(0)
