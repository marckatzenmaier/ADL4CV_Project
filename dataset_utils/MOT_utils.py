"""
@author Marc Katzenmaier
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


gt_labels = {'frame_nr': 0, 'box_id': 1, 'box_l': 2, 'box_t': 3, 'box_w': 4, 'box_h': 5, 'detection_conf': 6,
             'class_label': 7, 'visibility_ratio': 8}


class Mot17_info():
    """
    class contains all infos from the info file of the Mot GT
    """
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
    """
    :param path: path to a file with all image folders similar to Mot17_test_single.txt
    :return: list of all paths in the given file
    """
    file = open(path)
    paths = file.readlines()
    paths = list(map(lambda x:x[:-1], paths))
    file.close()
    return paths


def get_gt_info(path):
    """
    parse the gt and info file
    here the index is already fixed for python frame 0 is image 000001.jpg
    :param path: path to the folder of the sequence
    :return: tuple of (gt np.array of shape [N,9], Mot17_info()) where N is the number of boxes in the sequence
    """
    folder = ['gt/gt.txt', 'seqinfo.ini']
    info = Mot17_info(path+folder[1])
    gt = np.loadtxt(path+folder[0], delimiter=',')
    return convert_mat_py_notation(gt), info


def get_gt_img_inf(path):
    """
    Similar to get_gt_info(path) but also returns a list of paths to the individual images
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
    """
    :param gt: gt matrix with shape [N, 9]
    :param info: which value should be selected
    :param infotyp: which type should be selected need to be one of the keys of gt_labels
    :return: gt where the value of the infotype equals the info
    """
    return gt[gt[:, gt_labels[infotyp]] == info, :]

def filter_person(gt):
    """
    :param gt: gt matrix with shape [N, 9]
    :return: gt without any non person bounding boxes
    """
    index = gt[:, gt_labels['class_label']]
    index = ((index == 1).astype(np.int) + (index == 2).astype(np.int) + (index == 7).astype(np.int)) > 0
    return gt[index, :]

def filter_frames(gt, start, end):
    """
    :param gt: gt matrix with shape [N, 9]
    :param start: start frame number
    :param end: end frame number
    :return: returns all bounding boxes which fits to the images between start and end
    """
    index = gt[:, gt_labels['frame_nr']]
    index = np.logical_and((start <= index), (index < end))
    return gt[index, :]

def convert_mat_py_notation(gt):
    """
    Convertes the notation so that the first frame has index 0 as usual in python
    """
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

    """

    :param gt: Numpy array with shape (num_boxes, 4).
    :param height_factor: Factor to change height of boxes in gt.
    :param width_factor: Factor to change widht of boxes in gt.
    :return: Returns array with same shape as gt. Boxes are resized by given factors, but parameters are floats.
    """
    gt[:, [0, 2]] /= width_factor
    gt[:, [1, 3]] /= height_factor
    return gt

