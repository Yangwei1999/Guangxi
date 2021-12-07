
# yolo import

import numpy as np

from Yang_test.find_thed import double_check_HSV
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.plots import Annotator, colors, save_one_box, plot_one_box
from utils.torch_utils import select_device, time_sync
import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
def detector():
    # Load model
    half = False
    imgsz = [640,640]
    weights = 'best.pt'
    dnn = False
    device = select_device('0')
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    print('names',names)

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    print(imgsz)
    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    with torch.no_grad():
        a = os.listdir('image1')
        # a = ['05.png']
        for file_i in a:
            im = cv2.imread('image1'+'/'+file_i)
            img = im.copy()
            img_plot = im.copy()
            # Padded resize
            im = letterbox(im, imgsz, stride=stride, auto=pt and not jit)[0]
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            pred = model(im,augment=False,visualize=False)
            #NMS
            conf_thres = 0.25
            iou_thres = 0.45
            max_det = 1000
            classes= None
            agnostic_nms = None
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            for i,det in enumerate(pred):
                s= ''
                # Print results
                # det[:,-1].unique   return  a list that
                # img0 = cv2.imread(path)  # BGR
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()
                print(det)
                for *xyxy, conf, cls in reversed(det):
                    if(double_check_HSV(img,xyxy)>0.01):
                        plot_one_box(xyxy, img_plot, label=names[int(cls)], color=[255, 0, 0],
                                     line_thickness=2)
                cv2.imwrite('yang_res'+'/'+file_i,img_plot)







        # Process predictions
        # now the correspinding result has stored in pred, come back the picture photo








if __name__ == "__main__":
    detector()
    # im = cv2.imread('data/images/bus.jpg')
    # im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    # im = im.transpose(2,0,1)
    # im = np.ascontiguousarray(im)
    # print(im)
    # print(type(im))
    # im = torch.from_numpy(im)
    # print(im.shape)
