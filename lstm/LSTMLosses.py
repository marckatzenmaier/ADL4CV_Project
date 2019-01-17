import numpy as np
import torch
import torch.nn as nn
import torch.functional as f
import os


def prediction_to_box_list(pred_sequence, valid=False):
    box_list = []
    for pred in pred_sequence:
        # goal (batch_size, 120, boxes) boxes: (5) id, x, y, w, h
        input, div, target = pred


        batch_size = input.shape[0]

        # todo handle the case with vanishing boxes
        # todo idea all valid boxes are contained in target -> if a box vanishes it is not in target !! use target for
        # todo valid boxes

        # input wieder de normalisieren mach ich nicht hier
        # pred shape: (batch, grid, grid, anchor, 4)
        output = input.copy()
        output[:, :, :, :, 1:] += div
        output[:, :, :, :, 0] = target[:, :, :, :, 0]
        output[output[:, :, :, :, 0] == 0] = 0
        if valid:
            print("prediction")
            print(div[output[:, :, :, :, 0] != 0])
            print("truth")
            print(target[output[:, :, :, :, 0] != 0] - input[output[:, :, :, :, 0] != 0])

        output = output.reshape(batch_size, -1, 5)
        input = input.reshape(batch_size, -1, 5)
        target = target.reshape(batch_size, -1, 5)
        box_list.append((input, output, target))
    return box_list


def displacement_error(pred_box_list, metric):
    # take last element of sequence and compare them
    # shape (batch_size, 120, boxes) boxes: (5) id, x, y, w, h
    input, output, target = pred_box_list[-1]

    error = []
    for batch_id in range(len(target)):
        for box, box_id in enumerate(target[batch_id, :, 0]):
            if box_id != 0:
                target_box = np.where(output[batch_id, :, 0] == box_id)
                if len(target_box[0]) == 0:
                    print(np.unique(output[batch_id, :, 0]))
                    print(np.unique(input[batch_id, :, 0]))
                    print(np.unique(target[batch_id, :, 0]))
                target_box = output[batch_id, target_box[0][0], 1:] * 416 / 16.0
                error.append(metric(target_box, target[batch_id, box, 1:] * 416 / 16.0))
    return np.mean(error)


def mean_squared_trajectory_error(pred_box_list, metric):
    # take last element of sequence and compare them
    # shape (batch_size, 120, boxes) boxes: (5) id, x, y, w, h
    _, output, target = pred_box_list[-1]

    error = []
    for frame in pred_box_list:
        _, output, target = frame
        for batch_id in range(len(target)):
            for box, box_id in enumerate(target[batch_id, :, 0]):
                target_box = np.where(output[batch_id, :, 0] == box_id)
                target_box = output[batch_id, target_box, 1:]
                error.append(metric(target_box, target[batch_id, box, 1:]))

#def iou(box_1, box_2):
#
#    intersections = dx * dy
#    areas = (x2 - x1) * (y2 - y1)
#    unions = (areas + areas.t()) - intersections
#    ious = intersections / unions


def center_distance(box_1, box_2):
    return np.sqrt(np.sum(np.square(box_1[:2] - box_2[:2])))


class NaiveLoss(nn.modules.loss._Loss):

    def __init__(self, params):
        self.grid_shape = params["grid_shape"]  # (w h)
        self.image_shape = params["image_shape"]

        self.anchors = self.load_anchors(params["path_anchors"])
        self.num_anchors = len(self.anchors)

    def forward(self, pred, input, target):
        # calc mask because only cells/anchors with boxes are penalized
        # mask shape (batch_size, gridy, gridx, anchor)
        # todo handle the case with vanishing boxes
        # todo idea all valid boxes are contained in target -> if a box vanishes it is not in target !! use target for
        # todo valid boxes
        mask = target[:, :, :, :, 0] != 0
        mask = torch.unsqueeze(mask.float(), 4)

        input = input[:, :, :, :, 1:]
        target = target[:, :, :, :, 1:]
        #print(torch.masked_select(pred, (pred.detach() * mask.detach() > 0).byte()))
        return torch.sum(mask * torch.pow((target - input - pred), 2)) / torch.nonzero(mask).size(0)

    def to_yolo(self, input, target, thresh=0.0):
        """
        :param input: array with shape (batchsize, 120, 5)
        :param target: same shape as input
        :param thresh: threshold which determines which boxes are valid (0 if we use labels and 0.5 if we use predicted
                        boxes from yolo
        return: shape (batchsize, grid_h, grid_w, anchors, (id, x, y, w, h)) 
                x, y, w, h normed to -> x,y,w,h \in [0, grid_(w,h)]
        """
        # first the input
        batch_size = input.shape[0]
        grid_in = np.zeros([batch_size, self.grid_shape[1], self.grid_shape[0], self.num_anchors, 5])

        normed_input = np.zeros_like(input)
        normed_input[:, :, [1, 3]] = input[:, :, [1, 3]] * self.grid_shape[0] / self.image_shape[0]
        normed_input[:, :, [2, 4]] = input[:, :, [2, 4]] * self.grid_shape[1] / self.image_shape[1]
        normed_input[:, :, 0] = input[:, :, 0]

        cell_coord = np.floor(normed_input[:, :, [1, 2]]).astype(np.int)  # shape: 2 120 2
        valid_boxes = np.where(np.sum(input, axis=2) > thresh)  # shape 2 120
        batch_idx = valid_boxes[0]
        box_idx = valid_boxes[1]
        # for the moment calc the anchor box with euclidean metric
        # anchors shape (num_anchors, 2)
        anchor_idx = np.argmin(  # shape: 2 120
                     np.sum(np.abs(np.expand_dims(normed_input[:, :, [3, 4]], axis=2) - self.anchors), axis=3), axis=2)
        grid_y_idx = cell_coord[batch_idx, box_idx, 1]
        grid_x_idx = cell_coord[batch_idx, box_idx, 0]
        grid_in[batch_idx, grid_y_idx, grid_x_idx, anchor_idx[valid_boxes]] = normed_input[valid_boxes]

        # now the target
        # todo handle case in which the pedestrian vanishes in next frame
        grid_target = np.zeros([batch_size, self.grid_shape[1], self.grid_shape[0], self.num_anchors, 5])

        normed_target = np.zeros_like(target)
        normed_target[:, :, [1, 3]] = target[:, :, [1, 3]] * self.grid_shape[0] / self.image_shape[0]
        normed_target[:, :, [2, 4]] = target[:, :, [2, 4]] * self.grid_shape[1] / self.image_shape[1]
        normed_target[:, :, 0] = target[:, :, 0]

        valid_boxes_target = np.where(np.sum(target, axis=2) > 0)  # shape 2 120
        batch_idx_target = valid_boxes_target[0]
        box_idx_target = valid_boxes_target[1]
        cell_coord_target = []
        anchor_idx_target = []
        for batch, idx in zip(batch_idx_target, normed_target[batch_idx_target, box_idx_target, 0]):
            t = np.where(idx == normed_input[batch, :, 0])
            cell_coord_target.append(cell_coord[batch, t[0]][0])
            anchor_idx_target.append(anchor_idx[batch, t[0]][0])
        cell_coord_target = np.array(cell_coord_target)
        anchor_idx_target = np.array(anchor_idx_target)

        grid_target[batch_idx_target, cell_coord_target[:,1], cell_coord_target[:,0], anchor_idx_target] = normed_target[valid_boxes_target]

        return grid_in, grid_target

    def shift(self, input, pred):
        # input and pred are torch tensors
        # mask pred, add pred transform back to (batch, 120, 5) transform to yolo
        batch_size = input.shape[0]

        input[:, :, :, :, 1:] += pred
        return input

    def load_anchors(self, path):
        # normalize to grid size
        with open(path) as file:
            line = file.readline().split(", ")  # list of x,y
            anchors = np.zeros([len(line), 2])
            for i, box in enumerate(line):
                box = box.split(",")
                box = list(map(float, box))
                anchors[i, 0] = box[0]
                anchors[i, 1] = box[1]
            return anchors * self.grid_shape / self.image_shape
