from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import dataset_utils.MOT_utils as motu
import numpy as np
import matplotlib.pyplot as plt


class MotBBSequence(Dataset):
    """
    dataset which loads each frame individual
    """

    def __init__(self, paths_file, loader=default_loader, seq_length=20, new_height=224, new_width=224, step=10):
        """
        inits of all file names and bounding boxes
        """
        # framerate is not constant 05 SDP has 14 fps  vs 02 SDP 30 fps -> could be a problem
        # the movement is also a problem especially if the movement changes for the prediction time
        # todo include images for debugging -> at the moment the images are too much
        # crop negative boxes (maybe not necessary)
        self.loader = loader
        paths = motu.parse_videos_file(paths_file)
        self.all_gt = np.zeros([0, 5])

        #            dict    list  list
        # tmp format video->frame->boxes with id
        videos = {i: [] for i in range(len(paths))}
        # stats stuff
        all_traj_lengths = np.zeros(1)
        all_displacements = np.zeros(1)

        for i, path in enumerate(paths):
            # gt format: [frame id 4 box parameter]
            #tmp debug until i download the dataset
            path = '../../' + path
            #tmp
            # get ground truth (gt)
            gt, info = motu.get_gt_info(path)
            num_frames = info.seqLength
            print("Framerate:{}".format(info.frameRate))

            # remove everything that is not a pedestrian
            cond = np.vstack([(gt[:, 7] == 1).reshape(1, -1), (gt[:, 7] == 7).reshape(1, -1)]).T
            gt = gt[np.where(np.any(cond, axis=1))]
            # pedestrian ids are now discontinuous which results in tougher stats calculation
            # -> calc correction and apply it
            ids = gt[:, 1]
            ids_diff = ids[1:]-ids[:-1]  # first try with convolve didn't work
            ids_diff[ids_diff != 0] -= 1   # remove natural discontinuity resulting from counting upwards
            correct = np.cumsum(ids_diff)  # cum
            gt[1:, 1] -= correct  # assume the first sample has correct id
            gt[:, 1] -= gt[0, 1] - 1  # the pedestrians diff may begin with n != 1

            # change negative boxes (dont know if it is really important)(remove it if not necessary anymore)
            # ( i really hope there are no to big boxes -> there are to big boxes (face palm))

            neg_ids_x = np.where(gt[:, 2] < 0)
            neg_ids_y = np.where(gt[:, 3] < 0)
            pos_ids_x = np.where(gt[:, 2] + gt[:, 4] >= info.imWidth)
            pos_ids_y = np.where(gt[:, 3] + gt[:, 5] >= info.imHeight)
            gt[neg_ids_x, 4] += gt[neg_ids_x, 2]  # if we move the top left corner into the image we must adapt the height
            gt[neg_ids_x, 2] = 0
            gt[neg_ids_y, 5] += gt[neg_ids_y, 3]  # same here (dont know if y<0 exists)
            gt[neg_ids_y, 3] = 0
            gt[pos_ids_x, 4] = info.imWidth - gt[pos_ids_x, 2] - 1  # equal would also be bad
            gt[pos_ids_y, 5] = info.imHeight - gt[pos_ids_y, 3] - 1


            # resize boxes and normalize data
            height_scale = info.imHeight / new_height
            width_scale = info.imWidth / new_width
            gt = motu.transform_bb_to_centered(gt)
            gt[:, [2, 4]] /= width_scale
            gt[:, [3, 5]] /= height_scale

            # calc stats i.e trajectory length (mean , std, max min)
            max_ped_id = int(np.max(gt[:, 1]))  # ids start at 1
            # for each id calc the number of sample
            traj_lengths = np.zeros(max_ped_id)
            displacments = np.zeros(max_ped_id)
            for ped_id in range(max_ped_id):
                ids = np.where(gt[:, 1] == ped_id+1)[0]
                # trajectory length
                traj_lengths[ped_id] = ids.shape[0]
                # displacement from begin to end
                start = gt[ids[0], 2:4]
                end = gt[ids[-1], 2:4]
                if np.sqrt(np.sum(np.square(start-end))) > 224:
                    print(start, end)
                displacments[ped_id] = np.sqrt(np.sum(np.square(start-end)))
            all_traj_lengths = np.concatenate([all_traj_lengths, traj_lengths])
            all_displacements = np.concatenate([all_displacements, displacments])

            # create structure
            videos[i] = [None for j in range(num_frames)]
            for j in range(num_frames):
                videos[i][j] = gt[np.where(gt[:, 0] == j)]

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

        # print stats
        all_traj_lengths = all_traj_lengths[1:]  # first element is dummy
        print("Mean trajectory length: {}".format(np.mean(all_traj_lengths)))
        print("Deviation of trajectory length: {}".format(np.std(all_traj_lengths)))
        print("Max trajectory length: {}".format(np.max(all_traj_lengths)))
        print("Min trajectory length: {}".format(np.min(all_traj_lengths)))
        all_displacements = all_displacements[1:]
        print("Mean displacement: {}".format(np.mean(all_displacements)))
        print("Deviation of displacements: {}".format(np.std(all_displacements)))
        print("Max displacement: {}".format(np.max(all_displacements)))
        print("Min displacement: {}".format(np.min(all_displacements)))

    def __getitem__(self, index):
        """
        Returns sequence with static lenght.
        :param index:
        :return: Numpy array with shape (seq_length, 120, 5) (most of the 120 entries will be zeroes)
        
        """
        # todo seq_length 20 means that the 20th sample is only used as a target
        return self.sequences[index][:-1], self.sequences[index][1:]

    def __len__(self):
        """
        len of the dataset
        """
        return len(self.sequences)


# test
test_data = MotBBSequence('../Mot17_test_single.txt')
# draw boxes in black image
image = np.zeros((224, 224), dtype=np.uint8)
test_sequence = test_data[0][0]
print(test_sequence.shape)
for i in range(19):
    boxes = test_sequence[i]
    for box in range(120):
        if np.sum(boxes[box, 1:]) != 0:
            width = int(boxes[box, 3])
            height = int(boxes[box, 4])

            top_left_x = boxes[box, 1] - width / 2
            top_left_y = boxes[box, 2] - height / 2

            image[int(top_left_y), int(top_left_x): int(top_left_x + width)] = 255
            image[int(top_left_y + height), int(top_left_x): int(top_left_x + width)] = 255
            image[int(top_left_y):int(top_left_y + height), int(top_left_x)] = 255
            image[int(top_left_y):int(top_left_y + height), int(top_left_x + width)] = 255

    plt.imshow(image)
    plt.show()





