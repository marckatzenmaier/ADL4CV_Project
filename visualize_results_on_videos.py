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
from yolo.yolo_LSTM import YoloLSTM
from yolo.loss import YoloLoss
import torchvision.transforms as transforms
from yolo.yolo_utils import *
from PIL import Image
from PIL.ImageDraw import ImageDraw
from yolo.yolo_net import Yolo
from yolo.yolo_utils import *
import time

from PIL import Image

from Object_Detection_Metrics.lib.BoundingBox import BoundingBox
from Object_Detection_Metrics.lib.BoundingBoxes import BoundingBoxes
from Object_Detection_Metrics.lib.Evaluator import *
from Object_Detection_Metrics.lib.utils import *


path = 'dataset_utils/datasets/MOT17/MOT17-02-SDP/'


def get_gts_and_images(path, frac_train=0.8, min_visibility=0.0, min_width=0, min_height=0):
    gt, img, info = motu.get_gt_img_inf(path)
    modified_length = int(info.seqLength * frac_train)

    used_index = [0, 2, 3, 4, 5]

    gt = gt[gt[:, 2] < info.imWidth]  # remove boxes out of the image
    gt = gt[gt[:, 3] < info.imHeight]  # remove boxes out of the image

    neg_ids_x = np.where(gt[:, 2] < 0)
    neg_ids_y = np.where(gt[:, 3] < 0)

    pos_ids_x = np.where(gt[:, 2] + gt[:, 4] >= info.imWidth)
    pos_ids_y = np.where(gt[:, 3] + gt[:, 5] >= info.imHeight)
    gt[neg_ids_x, 4] += gt[
    neg_ids_x, 2]  # if we move the top left corner into the image we must adapt the height
    gt = gt[gt[:, 4] > 0]  # make sure that width stayed positiv
    gt[neg_ids_x, 2] = 0
    gt[neg_ids_y, 5] += gt[neg_ids_y, 3]  # same here (dont know if y<0 exists)
    gt = gt[gt[:, 5] > 0]  # make sure that height stayed positiv
    gt[neg_ids_y, 3] = 0
    gt[pos_ids_x, 4] = info.imWidth - gt[pos_ids_x, 2] - 1  # equal would also be bad
    gt[pos_ids_y, 5] = info.imHeight - gt[pos_ids_y, 3] - 1

    gt = gt[gt[:, 8] > min_visibility, :]
    gt = gt[gt[:, 5] > min_height, :]
    gt = gt[gt[:, 4] > min_width, :]

    gt = motu.transform_bb_to_centered(gt)
    gt = motu.filter_person(gt)
    #gt = motu.filter_frames(gt, 0, modified_length)
    gt = gt[:, used_index]
    return gt, img, modified_length, info


# todo gt, seperate train/val videos
def show_results_of_video(run_fkt, path, frac_train=0.8, min_visibility=0.0, min_width=0, min_height=0, img_size=416):
    gt, img, train_len = get_gts_and_images(path, frac_train=frac_train, min_visibility=min_visibility, min_width=min_width, min_height=min_height)
    gt_train = motu.filter_frames(gt, 0, train_len)
    gt_eval = motu.filter_frames(gt, train_len, len(img))
    img_train = img[0:train_len]
    img_eval = img[train_len:-1]
    # reset network x.reset
    for img_path in img_eval:
        img = default_loader(img_path)
        # run network x.detect() contains forward and postprocessing returns boxes
        # append boxes
    # calc accuracy


    '''images = []
    for ix, img_path in enumerate(img):
        im = default_loader(img_path)
        width, height = im.size
        width_ratio = float(img_size) / width
        height_ratio = float(img_size) / height
        boxes = run_fkt(im)
        im = make_boxed_img(im, boxes, width_ratio, height_ratio)
        #im.resize((480, 240))
        images.append(im)
    return images'''

def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

def make_boxed_img_ltwh(img, boxes, width_ratio, height_ratio):
    if len(boxes) != 0:
        predictions = boxes[0]
        width, height = img.size
        for pred in predictions:
            xmin = int(max(pred[0] / width_ratio, 0))
            ymin = int(max(pred[1] / height_ratio, 0))
            xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
            ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
            imgdraw = ImageDraw(img)
            print(f'{xmin}  {xmax}  {ymin}  {ymax}')
            drawrect(imgdraw, [(xmin, ymin), (xmax, ymax)], outline=(255,50,50), width=8)
    return img



def fix_neg_boxparams(boxes, width, height):
    for boxs in boxes:
        pass
    pass

def get_ap(logits, gt, width, height, anchors, IOU_threshold=.5):
    evaluator = Evaluator()
    allBoundingBoxes = BoundingBoxes()
    for i in range(gt.shape[0]):
        bb = BoundingBox('0', 0, gt[i, 1], gt[i, 2], gt[i, 3], gt[i, 4], CoordinatesType.Absolute,
                     (width, height), bbType=BBType.GroundTruth, format=BBFormat.XYWH)
        allBoundingBoxes.addBoundingBox(bb)
    #    #print(i)
    box_logits = logits_to_box_params(logits, anchors)
    coords = box_logits.transpose(2, 3).contiguous().view(-1, 5)
    coords = coords.cpu().numpy()
    coords[:, [0, 2]] *=width
    coords[:, [1, 3]] *=height
    for i in range(coords.shape[0]):
        bb = BoundingBox('0', 0, coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3], CoordinatesType.Absolute,
                         (info.imWidth, info.imHeight), BBType.Detected, coords[i, 4], format=BBFormat.XYWH)
        allBoundingBoxes.addBoundingBox(bb)
    metricsPerClass = evaluator.GetPascalVOCMetrics(allBoundingBoxes, IOUThreshold=IOU_threshold,
                                                    method=MethodAveragePrecision.EveryPointInterpolation)
    return metricsPerClass[0]['AP']



# todo make real run fkt
def sample_run_fkt(im):
    return [[np.array([10.0, 10.0, 100.0, 100.0])]]

img_path1 = '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-05-SDP/img1/000836.jpg'
path = 'dataset_utils/datasets/MOT17/MOT17-05-SDP/'
out_path = '/home/marc/Downloads/test.avi'
snapshot_path = '/home/marc/Downloads/snapshots/yolo_lstm_fast/snapshot0013.tar'
pretrained_path = '/home/marc/Downloads/snapshots/yolo_lr_4_fast/snapshot0020.tar'
yolo_pretrained = '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/models/yolo80_coco.pt'
yolo_snapshot = pretrained_path
batch_size=1
#net = YoloLSTM(batch_size)
#net.cpu()
#net.cuda()
#net.lstm_part.reinit_lstm(batch_size)
#net.encoder.cuda()
#net.lstm_part.cuda()
#net.load_snapshot(snapshot_path)
#net.load_pretrained_weights(pretrained_path)

net = Yolo()
net.cuda()
#net.load_pretrained_weights(yolo_pretrained)
net.load_snapshot(yolo_snapshot)
img=default_loader(img_path1)
#boxes = net.predict_boxes(img)

gt, imgs, eval_limit, info = get_gts_and_images(path)

#allBoundingBoxes = BoundingBoxes()

video_n = 0
#a = time.time()
#for i in range(gt.shape[0]):
#    bb = BoundingBox(f'{video_n:02d}_{int(gt[i,0]):04d}', 0, gt[i, 1], gt[i, 2], gt[i, 3], gt[i, 4], CoordinatesType.Absolute,
#                     (info.imWidth, info.imHeight), bbType=BBType.GroundTruth, format=BBFormat.XYWH)
#    allBoundingBoxes.addBoundingBox(bb)
#    break

#print(f'time used for one video: {time.time()-a}')
a = time.time()
#print(len(imgs))
evaluator = Evaluator()
all_ap = []
for ix, img_path in enumerate(imgs):
    break
    #print(ix)
    #allBoundingBoxes = BoundingBoxes()
    #print((motu.filter_gt(gt, ix)).shape)
    t_gt = motu.filter_gt(gt, ix)
    #for i in range(t_gt.shape[0]):
    #    bb = BoundingBox(f'{video_n:02d}_{ix:04d}', 0, t_gt[i, 1], t_gt[i, 2], t_gt[i, 3], t_gt[i, 4], CoordinatesType.Absolute,
    #                 (info.imWidth, info.imHeight), bbType=BBType.GroundTruth, format=BBFormat.XYWH)
    #    allBoundingBoxes.addBoundingBox(bb)
    #    #print(i)
    img = default_loader(img_path)
    width, height = img.size
    net.eval()
    trans = transforms.Compose([transforms.Resize((net.image_size, net.image_size)), transforms.ToTensor()])
    sample = trans(img).unsqueeze(0)
    sample = Variable(torch.FloatTensor(sample))
    device = next(net.parameters()).device
    sample = sample.to(device)
    logits = net(sample)
    all_ap.append(get_ap(logits,t_gt, width, height, net.anchors, .5))
    #box_logits = net.predict_modified_logits(img)
    #coords1 = box_logits.transpose(2, 3).contiguous().view(-1, 5)
    #coords = coords1.cpu().numpy()


    #width, height = img.size
    #coords[:, [0, 2]] *=width
    #coords[:, [1, 3]] *=height

    #for i in range(coords.shape[0]):
    #    bb = BoundingBox(f'{video_n:02d}_{ix:04d}', 0, coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3], CoordinatesType.Absolute,
    #                     (info.imWidth, info.imHeight), BBType.Detected, coords[i, 4], format=BBFormat.XYWH)
    #    allBoundingBoxes.addBoundingBox(bb)
    #metricsPerClass = evaluator.GetPascalVOCMetrics(allBoundingBoxes, IOUThreshold=.5, method=MethodAveragePrecision.EveryPointInterpolation)
    #all_ap.append(metricsPerClass[0]['AP'])
    #ap =
print(np.mean(np.array(all_ap)))
    #if ix >10:
    #    break
print(f'time used for one video: {time.time()-a}')
#evaluator = Evaluator()
#a = time.time()
#for i in [.5]:#, .6, .7, .8, .9]:
#metricsPerClass = evaluator.GetPascalVOCMetrics(allBoundingBoxes, IOUThreshold=.5, method=MethodAveragePrecision.EveryPointInterpolation)
#    print(f"{i}: {metricsPerClass[0]['AP']}")
#print(f'time used for one video: {time.time()-a}')

gt, imgs, eval_limit, info = get_gts_and_images(path)
img_path1 = imgs[0]
img = default_loader(img_path1)
width, height = img.size
width_ratio = 416.0 / width
height_ratio = 416.0 / height
print( width, height)
#boxes = net.predict_boxes(default_loader(img_path1))
#i = make_boxed_img(img, boxes, width_ratio, height_ratio)
#i.show()
t_gt = motu.filter_gt(gt, 0)

#print(t_gt.shape)
#print(t_gt)
#i = make_boxed_img_ccwh(img, [t_gt[:,1:5]], 1, 1)
#img.show()

trans = transforms.Compose([transforms.Resize((net.image_size, net.image_size)), transforms.ToTensor()])
sample = trans(img).unsqueeze(0)
sample = Variable(torch.FloatTensor(sample))
device = next(net.parameters()).device
sample = sample.to(device)
logits = net(sample)
#boxes = logits_to_box_params(logits, net.anchors)

#boxes = boxes.transpose(2, 3).contiguous().view(-1, 5)
#boxes = boxes.cpu().numpy()[:,0:4]*416
#def draw_img(logits, img, img_size, anchors):
#    img = img.contiguous().cpu().numpy()
#    boxes = post_processing(logits, img_size, anchors, .25, .5)
#    img = img.transpose(1,2,0)*255
#    img = Image.fromarray(img.astype(np.uint8), 'RGB')
#    img = make_boxed_img_ccwh(img, boxes, 1, 1)
#    return img

img = draw_img(logits, sample[0], 416, net.anchors)
img.show()
img = np.array(img)

#cv2.imshow('test', img)
#cv2.waitKey()
#img.show()
#print(next(net.encoder.parameters()).device)
#print(next(net.lstm_part.parameters()).device)
#print(net.lstm_part.cell.device)
#ims = show_results_of_video(sample_run_fkt, path)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter(out_path, fourcc, 20.0, (1920, 1080))
#for img in ims:
#    cvImg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#    print(cvImg.shape)
#    out.write(cvImg)
#out.release()