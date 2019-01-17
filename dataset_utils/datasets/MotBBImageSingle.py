from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import dataset_utils.MOT_utils as motu
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy import misc
from dataset_utils.datasets.MotBBSequence import MotBBSequence
import torchvision.transforms as trans
from dataset_utils.datasets.data_augmentation import *
import cv2
from yolo.yolo_utils import filter_non_zero_gt

def draw_boxes_opencv(img, boxes):
    width = float(img.shape[1])
    height = float(img.shape[0])
    for i in range(boxes.shape[0]):
        cv2.rectangle(img,
                      (int((boxes[i, 0] - boxes[i, 2] / 2) * width), int((boxes[i, 1] - boxes[i, 3] / 2) * height)),
                      (int((boxes[i, 0] + boxes[i, 2] / 2) * width), int((boxes[i, 1] + boxes[i, 3] / 2) * height)),
                      (0, 255, 0), 1)
    return img

class MotBBImageSingle(MotBBSequence):
    """
    dataset which loads each frame individual
    """
    def __init__(self, paths_file, seq_length=1, new_height=416, new_width=416, step=1,
                 valid_ratio=0.2, use_only_first_video=False, loader=default_loader, augment=True):
        super(MotBBImageSingle, self).__init__(paths_file, seq_length, new_height, new_width, step,
                                                 valid_ratio, use_only_first_video)
        self.loader = loader

        self.augment = augment
        self.transf = Compose([Crop(), VerticalFlip(), HSVAdjust(), Resize(new_width)])
        self.transf_eval = Resize(new_width)
        self.is_training = True

    def __getitem__(self, index):
        """
        Returns sequence of boxes with static length and frames .
        :param index:
        :return: Numpy array with shape (seq_length, 120, 5) (most of the 120 entries will be zeroes)

        """
        #images = torch.zeros([self.seq_length, 3, self.im_size[0], self.im_size[1]], dtype=torch.float32)
        for i in range(self.seq_length):
            '''#image = misc.imread(self.frame_paths[index][i])
            #image = misc.imresize(image, self.im_size)
            #print(image.dtype)
            #images[i] = image.transpose([2, 0, 1])  # convert to c w h format
            #a =self.transf(self.loader(self.frame_paths[index][i]))
            #print(a.max())'''
            #images[i] = self.transf(self.loader(self.frame_paths[index][i]))
            image = cv2.imread(self.frame_paths[index][i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = filter_non_zero_gt(self.sequences[index][0].copy())[:, 1:5] # boxes in ccwh
            boxes[:, [0, 2]] /= float(self.im_size[1])
            boxes[:, [1, 3]] /= float(self.im_size[0])  # boxes ccwh normalised
            if self.augment and self.is_training:
                image, boxes = self.transf((image, boxes))
            else:
                image, boxes = self.transf_eval((image, boxes))

            boxes[:, [0, 2]] *= float(self.im_size[1])
            boxes[:, [1, 3]] *= float(self.im_size[0])

            lable_out = np.zeros((120, 4))
            lable_out[0:boxes.shape[0], :] = boxes
            lable_out = torch.from_numpy(lable_out)#.unsqueeze(0)
            image = torch.from_numpy(np.transpose(image.astype(np.float32), (2, 0, 1)))#.unsqueeze(0)
            #
            '''#img = image.copy()  # for testing the augmentation functions
            #cv2.imshow('0', cv2.cvtColor(draw_boxes_opencv(img, boxes), cv2.COLOR_RGB2BGR))
            #cv2.waitKey()
            #b1 = boxes.copy()
            #data = (image, boxes)
            #a = Resize(416)
            #image, boxes = a(data)
            #img = image.copy()
            #cv2.imshow('1', cv2.cvtColor(draw_boxes_opencv(img, boxes), cv2.COLOR_RGB2BGR))
            #cv2.waitKey()'''

            ''' implemente only seq leng 1'''
            #for idx in range(len(objects)):
            #    objects[idx][4] = self.class_ids.index(objects[idx][4])
            #if self.is_training:
            #    transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.image_size)])
            #else:
            #    transformations = Compose([Resize(self.image_size)])
            #image, objects = transformations((image, objects))
            #return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)

        return lable_out, image

    def get_image_paths(self, index):
        return self.frame_paths[index][:-1], self.frame_paths[index][1:]

    def __len__(self):
        """
        len of the dataset
        """
        return len(self.sequences)


if __name__ == "__main__":
    # debug
    os.chdir("D:/Nikita/Documents/Projekte/ADL4CV_Project")
    test_data = MotBBImageSequence('dataset_utils/Mot17_test_single.txt')
    # draw boxes in black image
    test_sequence = test_data[0][0]
    images = test_data[0][2]
    for i in range(19):
        boxes = test_sequence[i]
        image = images[i]
        # image = np.zeros((416, 416), dtype=np.uint8)
        for box in range(120):
            if np.sum(boxes[box, 1:]) != 0:
                width = int(boxes[box, 3])
                height = int(boxes[box, 4])

                top_left_x = boxes[box, 1] - width / 2
                top_left_y = boxes[box, 2] - height / 2

                image[:, int(top_left_y), int(top_left_x): int(top_left_x + width)] = 255
                image[:, int(top_left_y + height), int(top_left_x): int(top_left_x + width)] = 255
                image[:, int(top_left_y):int(top_left_y + height), int(top_left_x)] = 255
                image[:, int(top_left_y):int(top_left_y + height), int(top_left_x + width)] = 255

        plt.imshow(image.transpose([1, 2, 0]))
        plt.show()





