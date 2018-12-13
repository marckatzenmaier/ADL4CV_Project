
import dataset_utils.MOT_utils as motu

"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import glob
#import torch
import argparse
import pickle
import cv2
import numpy as np
from yolo.yolo_utils import *
#import Yolo_pytorch.src.utils
from yolo.yolo_net import Yolo
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from PIL import Image
from PIL.ImageDraw import ImageDraw

CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
           "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
           "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
           "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush"]


def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
    parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/whole_model_trained_yolo_coco")
    parser.add_argument("--input", type=str, default="test_images")
    parser.add_argument("--output", type=str, default="test_images")

    args = parser.parse_args()
    return args


class Opt_visualize(object):
    def __init__(self):
        self.image_size = 416
        self.useCuda = True  # for easy disabeling cuda
        self.model = None
        self.conf_threshold = 0.25
        self.nms_threshold = 0.5


def load_from_snapshot(path, model, optim=None):
    checkpoint = torch.load(path)
    model.eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    if optim is None:
        return model
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optim


def run_network(img, opt):
    width, height = img.size
    width_ratio = float(opt.image_size) / width
    height_ratio = float(opt.image_size) / height
    opt.model.eval()
    trans = transforms.Compose([transforms.Resize((opt.image_size, opt.image_size)), transforms.ToTensor()])
    sample = trans(img).unsqueeze(0)
    sample = Variable(torch.FloatTensor(sample))
    device = torch.device('cpu')
    if torch.cuda.is_available() and opt.useCuda:
        sample = sample.cuda()
        device = torch.device("cuda")
    opt.model.to(device)
    with torch.no_grad():
        logits = opt.model(sample)
        #print('logits done')
        predictions = post_processing(logits, opt.image_size, CLASSES, opt.model.anchors, opt.conf_threshold,
                                      opt.nms_threshold, useCuda=opt.useCuda)
        return predictions, width_ratio, height_ratio


def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)


def make_boxed_img(img, boxes, width_ratio, height_ratio):
    #print(len(boxes))
    if len(boxes) != 0:
        predictions = boxes[0]
        #print(predictions)
        width, height = img.size
        #print(height)
        #output_image = cv2.imread(img[0])
        #output_image = cv2.resize(output_image, (opt.image_size, opt.image_size))
        for pred in predictions:

            xmin = int(max(pred[0] / width_ratio, 0))
            ymin = int(max(pred[1] / height_ratio, 0))
            xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
            ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
            #color = colors[CLASSES.index(pred[5])]
            if height - 1 < ymax-ymin:
                continue
            imgdraw = ImageDraw(img)
            #imgdraw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(255,0,0))
            drawrect(imgdraw, [(xmin, ymin), (xmax, ymax)], outline=(255,50,50), width=8)
            #cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
            #text_size = cv2.getTextSize(pred[5] + ' : %.2f' % pred[4], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            #cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
            #cv2.putText(
            #    output_image, pred[5] + ' : %.2f' % pred[4],
            #    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
            #    (255, 255, 255), 1)
            #print("Object: {}, Bounding box: ({},{}) ({},{})".format(pred[5], xmin, xmax, ymin, ymax))
        #predictions=
        #to_img = transforms.ToPILImage()
        #to_img(img).show()
        return img



def load_img(path):
    return default_loader(path)


def save_img(path, img):
    pass


'''def test(opt):
    # load model
    opt.pre_trained_model_path = '/home/marc/Documents/projects/ADL4CV_project/models/coco2014_2017/trained_models/whole_model_trained_yolo_coco'
    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model = torch.load('/home/marc/Documents/projects/ADL4CV_project/models/coco2014_2017/trained_models/whole_model_trained_yolo_coco')
            #model = torch.load(opt.pre_trained_model_path)
        else:
            model = Yolo(80)
            #model.load_state_dict(torch.load('/home/marc/Documents/projects/ADL4CV_project/models/coco2014_2017/trained_models/whole_model_trained_yolo_coco'))
            model.load_state_dict(torch.load(opt.pre_trained_model_path))
    else:
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
        else:
            model = Yolo(80)
            model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage))
    model.eval()
    colors = pickle.load(open("Yolo_pytorch/src/pallete", "rb"))

    # run model
    gt, img, info = motu.get_gt_img_inf(motu.parse_videos_file('/home/marc/Documents/projects/ADL4CV_project/code/dataset_utils/Mot17_test_single.txt')[4])
    image = cv2.imread(img[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (opt.image_size, opt.image_size))
    height, width = image.shape[:2]
    image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
    image = image[None, :, :, :]
    width_ratio = float(opt.image_size) / width
    height_ratio = float(opt.image_size) / height
    data = Variable(torch.FloatTensor(image))
    if torch.cuda.is_available():
        data = data.cuda()
    with torch.no_grad():
        logits = model(data)
        predictions = post_processing(logits, opt.image_size, CLASSES, model.anchors, opt.conf_threshold,
                                      opt.nms_threshold)
    if len(predictions) != 0:
        predictions = predictions[0]
        output_image = cv2.imread(img[0])
        output_image = cv2.resize(output_image, (opt.image_size, opt.image_size))
        for pred in predictions:
            xmin = int(max(pred[0] / width_ratio, 0))
            ymin = int(max(pred[1] / height_ratio, 0))
            xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
            ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
            color = colors[CLASSES.index(pred[5])]
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
            text_size = cv2.getTextSize(pred[5] + ' : %.2f' % pred[4], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
            cv2.putText(
                output_image, pred[5] + ' : %.2f' % pred[4],
                (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 255), 1)
            print("Object: {}, Bounding box: ({},{}) ({},{})".format(pred[5], xmin, xmax, ymin, ymax))
        cv2.imshow('img output', output_image)
        cv2.waitKey()'''


if __name__ == "__main__":
    opt = Opt_visualize()
    opt.model = Yolo(0, anchors=[(6.88, 27.44), (11.93, 58.32), (19.90, 94.92), (40.00, 195.84), (97.96, 358.62)])
    opt.model = load_from_snapshot('/home/marc/Downloads/log_ADL4CV/training_with_batch/snapshot0020.tar', opt.model)
    opt.useCuda = False
    opt.nms_threshold = 0.6
    opt.conf_threshold = 0.5
    img_paths = ['/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-09-SDP/img1/000094.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-09-SDP/img1/000427.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-02-SDP/img1/000247.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-04-SDP/img1/000451.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-05-SDP/img1/000518.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-10-SDP/img1/000267.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-11-SDP/img1/000143.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-13-SDP/img1/000359.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-02-SDP/img1/000596.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-04-SDP/img1/001050.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-05-SDP/img1/000836.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-09-SDP/img1/000524.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-10-SDP/img1/000654.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-11-SDP/img1/000899.jpg',
                '/home/marc/Documents/projects/ADL4CV_project/ADL4CV_Project/dataset_utils/datasets/MOT17/MOT17-13-SDP/img1/000750.jpg']
    for i, img_path in enumerate(img_paths):
        img = load_img(img_path)
        predictions, width_ratio, height_ratio = run_network(img, opt)
        img = make_boxed_img(img, predictions, width_ratio, height_ratio)
        img.save('/home/marc/Documents/projects/ADL4CV_project/imgs/{}.jpg'.format(i))
