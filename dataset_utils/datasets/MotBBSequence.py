from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import dataset_utils.MOT_utils as motu
import numpy as np


class MotBBSequence(Dataset):
    """
    dataset which loads each frame individual
    """

    def __init__(self, paths_file, loader=default_loader, seq_length=20, new_height=224, new_width=224):
        """
        inits of all file names and bounding boxes
        """
        self.loader = loader
        paths = motu.parse_videos_file(paths_file)
        self.all_gt = np.zeros([0, 5])

        # tmp format video->frame->boxes with id
        videos = {i: [] for i in range(len(paths))}

        for i, path in enumerate(paths):
            print(i)
            # gt format: [frame id 4 box parameter]
            #tmp debug until i download the dataset
            path = '../../' + path
            #tmp
            # todo implement stats (mean ped trajectory length, average displacement from begin to end)
            gt, info = motu.get_gt_info(path)
            num_frames = info.seqLength
            height_scale = info.imHeight / new_height
            width_scale = info.imWidth / new_width
            gt = motu.transform_bb_to_centered(gt)
            gt = motu.resize_bb(gt, height_scale, width_scale)  # i hope that's right

            videos[i] = [[]] * num_frames
            for j in range(gt.shape[0]):
                print("{} / {}".format(j, gt.shape[0]))
                videos[i][int(gt[j, 0])].append(gt[j, 1:])
        # todo implement final dataset structure so -> create finite sequences with constant length s.t index -> seq

    def __getitem__(self, index):
        """
        return sequence with static length
        """
        return sample, target

    def __len__(self):
        """
        len of the dataset
        """
        return len(self.all_imagepaths)


# test
a = MotBBSequence('../Mot17_test_single.txt')