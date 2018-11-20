from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import dataset_utils.MOT_utils as motu
import numpy as np


class MotBBSequence(Dataset):
    """
    dataset which loads each frame individual
    """

    def __init__(self, paths_file, loader=default_loader, seq_length=20, new_height=224, new_width=224, step=10):
        """
        inits of all file names and bounding boxes
        """
        self.loader = loader
        paths = motu.parse_videos_file(paths_file)
        self.all_gt = np.zeros([0, 5])

        #            dict    list  list
        # tmp format video->frame->boxes with id
        videos = {i: [] for i in range(len(paths))}

        for i, path in enumerate(paths):
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
            gt[:, [2, 4]] /= width_scale
            gt[:, [3, 5]] /= height_scale

            videos[i] = [None for j in range(num_frames)]
            for j in range(num_frames):
                videos[i][j] = gt[np.where(gt[:, 0] == j)]
        # todo implement final dataset structure -> create finite sequences with constant length s.t index -> seq
        # [batch_size, seq_index, (id bb)]
        # first idea : assume each frame has at maximum 120 (calc it in stats)
        self.sequences = []
        for video in videos.keys():  # todo look up wether .keys returns sorted list
            video_len = len(videos[video])

            start_index = 0
            while True:
                sequence = np.zeros([seq_length, 120, 5])  # 120 boxes with 5 parameter
                if start_index + seq_length < video_len:
                    sub_sequence = videos[video][start_index: start_index + seq_length]
                    for j in range(seq_length):
                        sequence[j, :sub_sequence[j].shape[0], :] = sub_sequence[j][:, 1:6]
                    self.sequences.append(sequence)
                    start_index += step
                else:
                    print("finished video {} at index {}/{}".format(video, start_index+seq_length, video_len))
                    break
        print(len(self.sequences))

    def __getitem__(self, index):
        """
        return sequence with static length
        """
        # todo
        return sample, target

    def __len__(self):
        """
        len of the dataset
        """
        return len(self.all_imagepaths)


# test
a = MotBBSequence('../Mot17_test_single.txt')