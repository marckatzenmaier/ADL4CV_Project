
import dataset_utils.MOT_utils as motu
#import torch
import pickle
from yolo.yolo_utils import *
#import Yolo_pytorch.src.utils
from yolo.yolo_net import Yolo
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from PIL import Image
from PIL.ImageDraw import ImageDraw


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
    sample = trans(img).unsqueeze(0)#*255
    sample = Variable(torch.FloatTensor(sample))
    device = torch.device('cpu')
    if torch.cuda.is_available() and opt.useCuda:
        sample = sample.cuda()
        device = torch.device("cuda")
    opt.model.to(device)
    with torch.no_grad():
        logits = opt.model(sample)
        predictions = post_processing(logits, opt.image_size, opt.model.anchors, opt.conf_threshold,
                                      opt.nms_threshold, useCuda=opt.useCuda)

        return predictions, width_ratio, height_ratio


def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)


def make_boxed_img(img, boxes, width_ratio, height_ratio):
    if len(boxes) != 0:
        predictions = boxes[0]
        width, height = img.size
        for pred in predictions:
            xmin = int(max(pred[0] / width_ratio, 0))
            ymin = int(max(pred[1] / height_ratio, 0))
            xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
            ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
            imgdraw = ImageDraw(img)
            drawrect(imgdraw, [(xmin, ymin), (xmax, ymax)], outline=(255,50,50), width=8)
    return img

def load_img(path):
    return default_loader(path)

if __name__ == "__main__":
    opt = Opt_visualize()
    anchor = [(6.88, 27.44), (11.93, 58.32), (19.90, 94.92), (40.00, 195.84), (97.96, 358.62)]
    anchor[:] = [(x[0] / 32.0, x[1] / 32.0) for x in anchor]
    opt.model = Yolo(0, anchors=anchor)
    snapshot_path = '/home/marc/Downloads/snapshots/fixed_anchors/snapshot0010.tar'
    opt.model = load_from_snapshot(snapshot_path, opt.model)
    opt.useCuda = False
    opt.nms_threshold = 0.5
    opt.conf_threshold = 0.25
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
        img.save('/home/marc/Documents/projects/ADL4CV_project/imgs/10_{}.jpg'.format(i))
