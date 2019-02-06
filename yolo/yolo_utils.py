"""
@author: Marc Katzenmaier
"""
import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate
from PIL import Image
from PIL.ImageDraw import ImageDraw
from Object_Detection_Metrics.lib_s.Evaluator import *
from Object_Detection_Metrics.lib_s.utils import *


def custom_collate_fn(batch):
    """
    function to combine the individual images and their box parameter to a batch
    """
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    items[1] = list(items[1])
    return items


def logits_to_box_params(logits, anchors):
    num_anchors = len(anchors)
    anchors = torch.Tensor(anchors)
    if isinstance(logits, Variable):
        logits = logits.data

    if logits.dim() == 3:
        logits.unsqueeze_(0)

    batch = logits.size(0)
    h = logits.size(2)
    w = logits.size(3)

    # Compute xc,yc, w,h, box_score on Tensor
    lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w)
    lin_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().contiguous().view(h * w)
    anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1)
    anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1)
    device = logits.device
    if torch.cuda.is_available() and device.type == "cuda":
        lin_x = lin_x.cuda()
        lin_y = lin_y.cuda()
        anchor_w = anchor_w.cuda()
        anchor_h = anchor_h.cuda()

    logits = logits.view(batch, num_anchors, -1, h * w).clone()  # sets bounding boxes
    logits[:, :, 0, :] = logits[:, :, 0, :].sigmoid().add(lin_x).div(w)
    logits[:, :, 1, :] = logits[:, :, 1, :].sigmoid().add(lin_y).div(h)
    logits[:, :, 2, :] = logits[:, :, 2, :].exp().mul(anchor_w).div(w)
    logits[:, :, 3, :] = logits[:, :, 3, :].exp().mul(anchor_h).div(h)
    logits[:, :, 4, :] = logits[:, :, 4, :].sigmoid()
    return logits


def filter_box_params(box_logits, image_size, anchors, conf_threshold, nms_threshold):
    """
    takes box parameter and retruns only valid boxes based on confidence treshold and non maximum suppression treshold
     in box type ltwh
    :param box_logits: modified logits of the net
    :param image_size: size of the input image feed in to the net
    :param anchors: of the net
    :param conf_threshold: Threshold for detecting a bounding box
    :param nms_threshold: non maximum supression threshold based on the IoU
    :return: a list of numpy arrays of shape [num_bounding_boxes, 4]
    """
    num_anchors = len(anchors)
    batch = box_logits.size(0)
    h = box_logits.size(2)
    w = box_logits.size(3)


    score_thresh = box_logits[:, :, 4, :] > conf_threshold
    score_thresh_flat = score_thresh.view(-1)

    if score_thresh.sum() == 0:
        predicted_boxes = []
        for i in range(batch):
            predicted_boxes.append(torch.Tensor([]))
    else:
        coords = box_logits.transpose(2, 3)[..., 0:4]
        coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
        scores = box_logits[:, :, 4, :][score_thresh]
        detections = torch.cat([coords, scores[:, None]], dim=1)

        max_det_per_batch = num_anchors * h * w
        slices = [slice(max_det_per_batch * i, max_det_per_batch * (i + 1)) for i in range(batch)]
        det_per_batch = torch.IntTensor([score_thresh_flat[s].int().sum() for s in slices])
        split_idx = torch.cumsum(det_per_batch, dim=0)

        # Group detections per image of batch
        predicted_boxes = []
        start = 0
        for end in split_idx:
            predicted_boxes.append(detections[start: end])
            start = end
    #here are all boxes with conv>= tresh in predicted_boxes

    selected_boxes = []
    for boxes in predicted_boxes:  # for each batch
        if boxes.numel() == 0:
            return boxes
        a = boxes[:, :2]  # x,y
        b = boxes[:, 2:4]  # width, eight
        bboxes = torch.cat([a - b / 2, a + b / 2], 1)  # conversion from center to 2 punkte
        scores = boxes[:, 4]

        # Sort coordinates by descending score
        scores, order = scores.sort(0, descending=True)
        x1, y1, x2, y2 = bboxes[order].split(1, 1)

        # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
        dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp(min=0)
        dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp(min=0)

        # Compute iou
        intersections = dx * dy
        areas = (x2 - x1) * (y2 - y1)  # area of each bb
        unions = (areas + areas.t()) - intersections
        ious = intersections / unions

        # Filter based on iou (and class)
        conflicting = (ious > nms_threshold).triu(1)
        keep = conflicting.sum(0).byte()
        keep = keep.cpu()
        conflicting = conflicting.cpu()
        keep_len = len(keep) - 1
        for i in range(1, keep_len):
            if keep[i] > 0:
                keep -= conflicting[i]
        if torch.cuda.is_available():
            keep = keep.cuda()

        keep = (keep == 0)
        selected_boxes.append(boxes[order][keep[:, None].expand_as(boxes)].view(-1, 5).contiguous())
    final_boxes = []
    for boxes in selected_boxes:
        if boxes.dim() == 0:
            final_boxes.append([])
        else:
            final_boxes.append(boxes)
    return final_boxes


def post_processing(logits, image_size, anchors, conf_threshold, nms_threshold):
    """
    transforms the logits of the net to bounding box parameters
    :param logits: output of the net
    :param image_size: height/width of the image since it is square
    :param anchors: anchors of the net with which it was trained with
    :param conf_threshold: Threshold for detecting a bounding box
    :param nms_threshold: non maximum supression threshold based on the IoU
    :return: a list of numpy arrays of shape [num_bounding_boxes, 4]
    """
    return filter_box_params(logits_to_box_params(logits, anchors),
                             image_size, anchors, conf_threshold, nms_threshold)


def draw_img(logits, img, img_size, anchors, conf_threshold=.25, nms_threshold=.5):
    """
    drawes the boxes into the image
    :param logits: logits of the network
    :param img: original image
    :return: image with drawn boxes
    """
    img = img.contiguous().cpu().numpy()
    boxes = post_processing(logits, img_size, anchors, conf_threshold, nms_threshold)
    img = img.transpose(1,2,0)*255
    img = Image.fromarray(img.astype(np.uint8), 'RGB')
    img = make_boxed_img_ccwh(img, boxes, 1, 1)
    return img


def make_boxed_img_ccwh(img, boxes, width_ratio, height_ratio, width=416, height=416):
    """
    helpfunction for draw_img
    """
    if len(boxes) != 0:
        predictions = boxes[0]
        for pred in predictions:
            xmin = int(max((pred[0]) / width_ratio, 0))
            ymin = int(max((pred[1]) / height_ratio, 0))
            xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
            ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
            imgdraw = ImageDraw(img)
            drawrect(imgdraw, [(xmin, ymin), (xmax, ymax)], outline=(255, 50, 50), width=2)
    return img


def drawrect(drawcontext, xy, outline=None, width=2):
    """
    draw box with bigger line
    """
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)


def get_ap(logits, gt, width, height, anchors, IOU_threshold=.5):
    """
    returns the AP for the given gt and logits
    :param logits: output of the net
    :param gt: gt bounding boxes
    :param width: width of the image
    :param height: heigth of the image
    :param anchors: anchors of the net
    :param IOU_threshold: threshold for the AP metric when it is declared as true positiv
    :return: AP value
    """
    evaluator = Evaluator()
    allBoundingBoxes = BoundingBoxes()
    gt=gt.cpu().numpy()
    for i in range(gt.shape[0]):
        bb = BoundingBox('0', 0, gt[i, 0], gt[i, 1], gt[i, 2], gt[i, 3], CoordinatesType.Absolute,
                         (width, height), bbType=BBType.GroundTruth, format=BBFormat.XYWH)
        allBoundingBoxes.addBoundingBox(bb)
    coords = filter_box_params(logits_to_box_params(logits, anchors), width, anchors, 0., 0.5)[0]
    coords = coords.cpu().numpy()
    coords[:, [0, 2]] *= width
    coords[:, [1, 3]] *= height
    for i in range(coords.shape[0]):
        bb = BoundingBox('0', 0, coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3], CoordinatesType.Absolute,
                         (width, height), BBType.Detected, coords[i, 4], format=BBFormat.XYWH)
        allBoundingBoxes.addBoundingBox(bb)
    metricsPerClass = evaluator.GetPascalVOCMetrics(allBoundingBoxes, IOUThreshold=IOU_threshold,
                                                    method=MethodAveragePrecision.EveryPointInterpolation)
    return metricsPerClass[0]['AP']


def filter_non_zero_gt(gt):
    """used to filter only existing bb for loss"""
    return gt[gt[:, 0] != 0.0][:, 1:]


def filter_non_zero_gt_without_id(gt):
    """used to filter only existing bb for loss"""
    return gt[gt[:, 3] != 0.0]


def draw_boxes_opencv(img, boxes):
    """
    :param img: image in opencv format
    :param boxes: numpy array with bounding boxes of shape [Number_boxes, 4] (in format ccwh)
    :return: image with added boxes
    """
    width = float(img.shape[1])
    height = float(img.shape[0])
    for i in range(boxes.shape[0]):
        cv2.rectangle(img,
                      (int((boxes[i, 0] - boxes[i, 2] / 2) * width), int((boxes[i, 1] - boxes[i, 3] / 2) * height)),
                      (int((boxes[i, 0] + boxes[i, 2] / 2) * width), int((boxes[i, 1] + boxes[i, 3] / 2) * height)),
                      (0, 0, 255), 2)
    return img