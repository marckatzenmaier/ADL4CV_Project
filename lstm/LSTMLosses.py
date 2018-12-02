import numpy as np
import torch
import torch.nn as nn
import torch.functional as f
import os


class NaiveLoss(nn.modules.loss._Loss):

    def __init__(self, params):
        self.grid_shape = params["grid_shape"]  # (w h)
        self.image_shape = params["image_shape"]

        self.anchors = self.load_anchors(params["path_anchors"])
        self.num_anchors = len(self.anchors)

    def forward(self, pred, input, target):
        # calc mask because only cells/anchors with boxes are penalized
        # mask shape (batch_size, gridy, gridx, anchor)
        mask = (input[:, :, :, :, 0] != 0)
        mask = torch.unsqueeze(torch.Tensor(mask.astype(np.int)), 4)

        input = torch.Tensor(input[:, :, :, :, 1:])
        target = torch.Tensor(target[:, :, :, :, 1:])


        return torch.sum(mask * torch.pow((target - input + pred), 2))

    def to_yolo(self, input, target):
        """
        :param input: array with shape (batchsize, 120, 5)
        :param target: same shape as input
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
        valid_boxes = np.where(np.sum(input, axis=2) > 0)  # shape 2 120
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
        # mask pred, add pred transform back to (batch, 120, 5) transform to yolo
        mask = (input[:, :, :, :, 0] != 0)
        mask = mask.int()
        pred = mask * pred

        shifted = input + pred


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
