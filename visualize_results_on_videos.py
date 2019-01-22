from dataset_utils import MOT_utils as motu
import torch.nn as nn
import torch
import shutil
import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
from torchvision.datasets.folder import default_loader
from run_sample_yolo import make_boxed_img
import cv2
#from yolo.yolo_LSTM import YoloLSTM
from yolo.loss import YoloLoss
import torchvision.transforms as transforms
from yolo.yolo_utils import *
from PIL import Image
from PIL.ImageDraw import ImageDraw
from yolo.yolo_net import Yolo
from yolo.yolo_utils import *
import time

from PIL import Image

from Object_Detection_Metrics.lib_s.BoundingBox import BoundingBox
from Object_Detection_Metrics.lib_s.BoundingBoxes import BoundingBoxes
from Object_Detection_Metrics.lib_s.Evaluator import *
from Object_Detection_Metrics.lib_s.utils import *



def draw_boxes_opencv(img, boxes):
    width = float(img.shape[1])
    height = float(img.shape[0])
    for i in range(boxes.shape[0]):
        cv2.rectangle(img,
                      (int((boxes[i, 0] - boxes[i, 2] / 2) * width), int((boxes[i, 1] - boxes[i, 3] / 2) * height)),
                      (int((boxes[i, 0] + boxes[i, 2] / 2) * width), int((boxes[i, 1] + boxes[i, 3] / 2) * height)),
                      (0, 255, 0), 1)
    return img

path = 'dataset_utils/datasets/MOT17/MOT17-02-SDP/'




#img_path1 = '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-05-SDP/img1/000836.jpg'
path = 'dataset_utils/datasets/MOT17/MOT17-13-SDP/'
out_path = '/home/marc/Downloads/13.avi'
snapshot_path = '/home/marc/Downloads/snapshots/yolo_832/snapshot0025.tar'
#pretrained_path = '/home/marc/Downloads/snapshots/yolo_lr_4_fast/snapshot0020.tar'
#yolo_pretrained = '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/models/yolo80_coco.pt'
#yolo_snapshot = pretrained_path
batch_size=1
anchors=[(0.215*2, 0.8575*2), (0.3728125*2, 1.8225*2), (0.621875*2, 2.96625*2), (1.25*2, 6.12*2), (3.06125*2, 11.206875*2)]
net = Yolo(image_size=832, anchors=anchors)
net.load_snapshot(snapshot_path)
net.eval()
net.cuda()
imgs = []
_, img_paths, _ = motu.get_gt_img_inf(path)
for img_path in img_paths[int(len(img_paths)*.8):]:
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (net.image_size, net.image_size))
    img = image.copy()
    image = torch.from_numpy(np.transpose(image.astype(np.float32), (2, 0, 1))).unsqueeze(0).cuda()
    boxes = post_processing(net(image), net.image_size, net.anchors, 0.25, 0.5)
    img = draw_boxes_opencv(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), boxes[0])
    imgs.append(img)
    #cv2.imshow('img', img)
    #cv2.waitKey(20)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_path, fourcc, 20.0, (832, 832))
for i in imgs:
    out.write(i)
out.release()