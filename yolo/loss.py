"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn


class YoloLoss(nn.modules.loss._Loss):
    # The loss I borrow from LightNet repo.
    def __init__(self, num_classes, anchors, reduction=32, coord_scale=1.0, noobject_scale=1.0,
                 object_scale=5.0, class_scale=1.0, thresh=0.6):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchor_step = len(anchors[0])
        self.anchors = torch.Tensor(anchors)
        self.reduction = reduction

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.thresh = thresh

    def forward(self, output, target):
        #print(output.shape)
        #print(target.shape)
        batch_size = output.data.size(0)
        height = output.data.size(2)
        width = output.data.size(3)

        # Get x,y,w,h,conf,cls
        output = output.view(batch_size, self.num_anchors, -1, height * width)
        coord = torch.zeros_like(output[:, :, :4, :])
        coord[:, :, :2, :] = output[:, :, :2, :].sigmoid()  # center coordinate with respect ot box
        coord[:, :, 2:4, :] = output[:, :, 2:4, :]  # scaling factor for anchor
        conf = output[:, :, 4, :].sigmoid()  # confidence
        #cls = output[:, :, 5:, :].contiguous().view(batch_size * self.num_anchors, self.num_classes,
        #                                            height * width).transpose(1, 2).contiguous().view(-1,
        #                                                                                              self.num_classes)
        #print(output[:, :, 2:4, :].max())
        #print(output[:, :, :2, :].max())
        # Create prediction boxes
        pred_boxes = torch.FloatTensor(batch_size * self.num_anchors * height * width, 4)
        lin_x = torch.range(0, width - 1).repeat(height, 1).view(height * width)
        lin_y = torch.range(0, height - 1).repeat(width, 1).t().contiguous().view(height * width)
        anchor_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)

        if torch.cuda.is_available() and output.is_cuda:
            pred_boxes = pred_boxes.cuda()
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)
        pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)
        pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
        pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)
        pred_boxes = pred_boxes.cpu()

        # Get target values
        coord_mask, conf_mask, tcoord, tconf = self.build_targets(pred_boxes, target, height, width)
        coord_mask = coord_mask.expand_as(tcoord)

        if torch.cuda.is_available() and output.is_cuda:
            tcoord = tcoord.cuda()
            tconf = tconf.cuda()
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()

        conf_mask = conf_mask.sqrt()
        #print(coord_mask.shape)
        #print(coord.shape)
        #print(tcoord.shape)
        # Compute losses
        mse = nn.MSELoss(size_average=False)
        self.loss_coord = self.coord_scale * mse(coord * coord_mask, tcoord * coord_mask) / batch_size
        self.loss_conf = mse(conf * conf_mask, tconf * conf_mask) / batch_size
        self.loss_tot = self.loss_coord + self.loss_conf

        return self.loss_tot, self.loss_coord, self.loss_conf

    def build_targets(self, pred_boxes, ground_truth, height, width):
        batch_size = len(ground_truth)

        conf_mask = torch.ones(batch_size, self.num_anchors, height * width, requires_grad=False) * self.noobject_scale
        coord_mask = torch.zeros(batch_size, self.num_anchors, 1, height * width, requires_grad=False)
        tcoord = torch.zeros(batch_size, self.num_anchors, 4, height * width, requires_grad=False)
        tconf = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)

        for b in range(batch_size):

            if len(ground_truth[b]) == 0:
                continue

            # Build up tensors
            cur_pred_boxes = pred_boxes[
                             b * (self.num_anchors * height * width):(b + 1) * (self.num_anchors * height * width)]
            if self.anchor_step == 4:
                anchors = self.anchors.clone()
                anchors[:, :2] = 0
            else:
                anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)
            #gt = torch.zeros(len(ground_truth[b]), 4)

            # todo gt as torch tensor
            # todo remove since dataset is already correct formated and so multiple calls will break the system
            #ground_truth[b, :, 0] = ground_truth[b, :, 0] +ground_truth[b, :, 2]/2
            #ground_truth[b, :, 1] = ground_truth[b, :, 1] +ground_truth[b, :, 3]/2
            gt = (ground_truth[b]/self.reduction).type(torch.FloatTensor)

            """for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = (anno[0] + anno[2] / 2) / self.reduction
                gt[i, 1] = (anno[1] + anno[3] / 2) / self.reduction
                gt[i, 2] = anno[2] / self.reduction
                gt[i, 3] = anno[3] / self.reduction"""

            # Set confidence mask of matching detections to 0
            iou_gt_pred = bbox_ious(gt, cur_pred_boxes)
            mask = (iou_gt_pred > self.thresh).sum(0) >= 1
            conf_mask[b][mask.view_as(conf_mask[b])] = 0

            # Find best anchor for each ground truth
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = bbox_ious(gt_wh, anchors)
            _, best_anchors = iou_gt_anchors.max(1)

            # Set masks and target values for each ground truth
            '''
            # unvectorized code 10 times slower then vectorized
            for i, anno in enumerate(ground_truth[b]):
                gi = min(width - 1, max(0, int(gt[i, 0])))
                gj = min(height - 1, max(0, int(gt[i, 1])))
                best_n = best_anchors[i]
                iou = iou_gt_pred[i][best_n * height * width + gj * width + gi]
                coord_mask[b][best_n][0][gj * width + gi] = 1
                #cls_mask[b][best_n][gj * width + gi] = 1
                conf_mask[b][best_n][gj * width + gi] = self.object_scale
                tcoord[b][best_n][0][gj * width + gi] = gt[i, 0] - gi
                tcoord[b][best_n][1][gj * width + gi] = gt[i, 1] - gj
                tcoord[b][best_n][2][gj * width + gi] = math.log(max(gt[i, 2], 1.0) / self.anchors[best_n, 0])
                tcoord[b][best_n][3][gj * width + gi] = math.log(max(gt[i, 3], 1.0) / self.anchors[best_n, 1])
                tconf[b][best_n][gj * width + gi] = iou
                #tcls[b][best_n][gj * width + gi] = int(anno[4])'''

            gij = gt.clone().long()
            gij[:, 0].clamp_(min=0, max=width-1)
            gij[:, 1].clamp_(min=0, max=height-1)
            iou = iou_gt_pred[range(len(ground_truth[b])), best_anchors * height * width + gij[:, 1] * width + gij[:, 0]]
            coord_mask[b, best_anchors, 0, gij[:, 1] * width + gij[:, 0]] = 1
            conf_mask[b, best_anchors, gij[:, 1] * width + gij[:, 0]] = self.object_scale

            tcoord[b, best_anchors, 0, gij[:, 1] * width + gij[:, 0]] = gt[:, 0].float() - gij[:, 0].float()
            tcoord[b, best_anchors, 1, gij[:, 1] * width + gij[:, 0]] = gt[:, 1].float() - gij[:, 1].float()

            tcoord[b, best_anchors, 2, gij[:, 1] * width + gij[:, 0]] = \
                torch.log(gt[:, 2].clamp(min=1.0) / self.anchors[best_anchors, 0])
            tcoord[b, best_anchors, 3, gij[:, 1] * width + gij[:, 0]] = \
                torch.log(gt[:, 3].clamp(min=1.0) / self.anchors[best_anchors, 1])

            tconf[b, best_anchors, gij[:, 1] * width + gij[:, 0]] = iou

        return coord_mask, conf_mask, tcoord, tconf


def bbox_ious(boxes1, boxes2):
    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions
