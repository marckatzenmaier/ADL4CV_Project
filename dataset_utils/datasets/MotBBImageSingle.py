"""
@author Marc Katzenmaier
"""
from torchvision.datasets.folder import default_loader
import torch
from dataset_utils.datasets.MotBBSequence import MotBBSequence
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
        for i in range(self.seq_length):
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
            lable_out = torch.from_numpy(lable_out)
            image = torch.from_numpy(np.transpose(image.astype(np.float32), (2, 0, 1)))

        return lable_out, image

    def get_image_paths(self, index):
        return self.frame_paths[index][:-1], self.frame_paths[index][1:]

    def __len__(self):
        """
        len of the dataset
        """
        return len(self.sequences)