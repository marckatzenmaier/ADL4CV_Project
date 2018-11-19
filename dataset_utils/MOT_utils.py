import numpy as np
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import scipy
import matplotlib.image as mpimg


# todo labels for 6,7,8
gt_labels = {'frame_nr': 0, 'box_id': 1, 'box_l': 2, 'box_t': 3, 'box_w': 4, 'box_h': 5, 'a': 6, 'b': 7, 'c': 8}


class Mot17_info():
    def __init__(self, path):
        file = open(path)
        infos = file.readlines()
        file.close()
        self.name = infos[1].split('=', 1)[-1]
        self.frameRate = int(infos[3].split('=', 1)[-1])
        self.seqLength = int(infos[4].split('=', 1)[-1])
        self.imWidth = int(infos[5].split('=', 1)[-1])
        self.imHeight = int(infos[6].split('=', 1)[-1])
        self.imExt = infos[7].split('=', 1)[-1]


def parse_videos_file(path):
    file = open(path)
    paths = file.readlines()
    paths = list(map(lambda x:x[:-1], paths))
    file.close()
    return paths


def get_gt_img_inf(path):
    """
        here the index is already fixed for python frame 0 is image 000001.jpg
    """
    folder = ['gt/gt.txt', 'img1/', 'seqinfo.ini']
    imgs = os.listdir(path+folder[1])
    imgs = sorted(imgs)
    imgs = list(map(lambda x: path+folder[1]+x, imgs))
    info = Mot17_info(path+folder[2])
    gt = np.loadtxt(path+folder[0], delimiter=',')
    return convert_mat_py_notation(gt), imgs, info


def filter_gt(gt, info, infotyp='frame_nr'):
    return gt[gt[:, gt_labels[infotyp]] == info, :]


def convert_mat_py_notation(gt):
    gt_py = gt
    gt_py[:, [0, 2, 3]] -= 1
    return gt_py


def loadImg(path):
    return mpimg.imread(path)


def imshow(img):
    plt.imshow(img)


def transform_bb_to_centered(gt):
    """
    for the original gt data:[framenr, id, left, top, width, height]

    """
    gt[:, gt_labels['box_l']] = gt[:, gt_labels['box_l']] + gt[:, gt_labels['box_w']]/2
    gt[:, gt_labels['box_t']] = gt[:, gt_labels['box_t']] + gt[:, gt_labels['box_h']]/2
    return gt


def resize_bb(gt, height_factor, width_factor):
    gt[:, [0, 2]] /= width_factor
    gt[:, [1, 3]] /= height_factor
    return gt


#testing stuff
# todo care 0<>1 origin for image list and pixel origin
# print(parse_videos_file('Mot17_test_single.txt')[0])
#gt, img, info = get_gt_img_inf(parse_videos_file('Mot17_test_single.txt')[0])
#plt.imshow(loadImg(img[0]))
#plt.show()
