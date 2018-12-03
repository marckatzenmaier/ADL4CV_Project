from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import dataset_utils.MOT_utils as motu
import numpy as np
import torchvision.transforms as trans
import torch

class MOT_bb_singleframe(Dataset):
    """
    dataset which loads each frame individual
    """
    def __init__(self, paths_file, loader=default_loader, transform=None, target_transform=None, frac_train=0.8):
        """
        inits of all file names and bounding boxes
        """
        self.loader = loader
        self.transform = trans.Compose([transform, trans.ToTensor()])
        self.target_transform = target_transform
        paths = motu.parse_videos_file(paths_file)
        current_index = 0
        used_index = [0, 2, 3, 4, 5]
        self.all_gt = np.zeros([0, 5])
        self.all_imagepaths = []
        self.num_classes = 80  # todo remove its only for first test with loss requires class

        for path in paths:
            gt, img, info = motu.get_gt_img_inf(path)
            modified_length = int(info.seqLength * frac_train)
            gt = motu.transform_bb_to_centered(gt)
            gt = motu.filter_person(gt)
            gt = motu.filter_frames(gt, 0, modified_length)
            gt = gt[:, used_index]
            gt[:, 0] += current_index
            current_index += modified_length
            self.all_gt = np.concatenate((self.all_gt, gt), 0)
            self.all_imagepaths += img[:modified_length]

    def __getitem__(self, index):
        """
        load the image itself
        and choose the right bb frame
        """
        target = motu.filter_gt(self.all_gt, index)
        target = torch.from_numpy(target[:, 1:])
        sample = self.loader(self.all_imagepaths[index])
        width, height = sample.size
        if self.transform is not None:
            sample = self.transform(sample)
        if height != sample.shape[1] and width != sample.shape[2]:
            height_scale = height/sample.shape[1]
            width_scale = width/sample.shape[1]
            if self.target_transform is not None:
                target = trans.Compose([trans.Lambda(lambda x: motu.resize_bb(x, height_scale, width_scale)),
                                        self.target_transform])(target)
            else:
                target = motu.resize_bb(target, height_scale, width_scale)
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        """
        len of the dataset
        """
        return len(self.all_imagepaths)


class MOT_bb_singleframe_eval(Dataset):
    """
    dataset which loads each frame individual
    """
    def __init__(self, paths_file, loader=default_loader, transform=None, target_transform=None, frac_train=0.8):
        """
        inits of all file names and bounding boxes
        """
        self.loader = loader
        self.transform = trans.Compose([transform, trans.ToTensor()])
        self.target_transform = target_transform
        paths = motu.parse_videos_file(paths_file)
        current_index = 0
        used_index = [0, 2, 3, 4, 5]
        self.all_gt = np.zeros([0, 5])
        self.all_imagepaths = []
        self.num_classes = 80  # todo remove its only for first test with loss requires class

        for path in paths:
            gt, img, info = motu.get_gt_img_inf(path)
            modified_length = int(info.seqLength * frac_train)
            gt = motu.transform_bb_to_centered(gt)
            gt = motu.filter_person(gt)
            gt = motu.filter_frames(gt, modified_length, info.seqLength)
            gt = gt[:, used_index]
            gt[:, 0] -= modified_length
            gt[:, 0] += current_index
            current_index += info.seqLength-modified_length
            self.all_gt = np.concatenate((self.all_gt, gt), 0)
            self.all_imagepaths += img[modified_length:]

    def __getitem__(self, index):
        """
        load the image itself
        and choose the right bb frame
        """
        target = motu.filter_gt(self.all_gt, index)
        target = torch.from_numpy(target[:, 1:])
        sample = self.loader(self.all_imagepaths[index])
        width, height = sample.size
        if self.transform is not None:
            sample = self.transform(sample)
        if height != sample.shape[1] and width != sample.shape[2]:
            height_scale = height/sample.shape[1]
            width_scale = width/sample.shape[1]
            if self.target_transform is not None:
                target = trans.Compose([trans.Lambda(lambda x: motu.resize_bb(x, height_scale, width_scale)),
                                        self.target_transform])(target)
            else:
                target = motu.resize_bb(target, height_scale, width_scale)
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        """
        len of the dataset
        """
        return len(self.all_imagepaths)

