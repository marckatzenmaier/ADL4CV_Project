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


class MotBBImageSequence(MotBBSequence):
    """
    dataset which loads each frame individual
    """
    def __init__(self, paths_file, seq_length=20, new_height=416, new_width=416, step=5,
                 valid_ratio=0.2, use_only_first_video=False, loader=default_loader, transform=None):
        super(MotBBImageSequence, self).__init__(paths_file, seq_length, new_height, new_width, step,
                                                 valid_ratio, use_only_first_video)
        self.loader = loader
        if transform is not None:
            self.transf = trans.Compose([trans.Resize((new_height, new_width)), transform, trans.ToTensor()])
        else:
            self.transf = trans.Compose([trans.Resize((new_height, new_width)), trans.ToTensor()])


    def __getitem__(self, index):
        """
        Returns sequence of boxes with static length and frames .
        :param index:
        :return: Numpy array with shape (seq_length, 120, 5) (most of the 120 entries will be zeroes)

        """
        images = torch.zeros([self.seq_length, 3, self.im_size[0], self.im_size[1]], dtype=torch.float32)
        for i in range(self.seq_length):
            '''#image = misc.imread(self.frame_paths[index][i])
            #image = misc.imresize(image, self.im_size)
            #print(image.dtype)
            #images[i] = image.transpose([2, 0, 1])  # convert to c w h format
            #a =self.transf(self.loader(self.frame_paths[index][i]))
            #print(a.max())'''
            images[i] = self.transf(self.loader(self.frame_paths[index][i]))


        # todo seq_length 20 means that the 20th sample is only used as a target
        return self.sequences[index][:-1], self.sequences[index][1:], images[:-1], images[1:]

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





