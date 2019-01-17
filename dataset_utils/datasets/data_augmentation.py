import numpy as np
from random import uniform
import cv2

''' all functions take as input data (image, gt_boxes) where the boxes need to be in ccwh format
 and normalized by the image width and height, it will be returned the same format'''
class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for function_ in self.transforms:
            data = function_(data)
        return data


class Crop(object):
    # fixed und  tested
    def __init__(self, max_crop=0.6, prob=0.5):
        super().__init__()
        self.max_crop = max_crop
        self.prob = prob

    def __call__(self, data):
        image, label = data
        if uniform(0, 1) < self.prob:
            height, width = image.shape[:2]
            temp_w = uniform(1-self.max_crop, 1)
            temp_h = uniform(1-self.max_crop, 1)
            cropped_left = uniform(0, 1-temp_w)
            cropped_top = uniform(0, 1-temp_h)
            cropped_right = 1 - temp_w - cropped_left
            cropped_bottom = 1 - temp_h - cropped_top

            # convert ccwh to ltrb
            label_ltrb = label.copy()
            label_ltrb[:, 0] = label[:, 0] - label[:, 2]/2
            label_ltrb[:, 1] = label[:, 1] - label[:, 3]/2
            label_ltrb[:, 2] = label[:, 0] + label[:, 2]/2
            label_ltrb[:, 3] = label[:, 1] + label[:, 3]/2
            # modifie label
            label_ltrb[:, [0, 2]] -= cropped_left
            label_ltrb[:, [1, 3]] -= cropped_top
            label_ltrb[label_ltrb[:, 0] > 1 - cropped_right - cropped_left, 0] = 1 - cropped_right - cropped_left
            label_ltrb[label_ltrb[:, 1] > 1 - cropped_bottom - cropped_top, 1] = 1 - cropped_bottom - cropped_top
            label_ltrb[label_ltrb[:, 2] > 1 - cropped_right - cropped_left, 2] = 1 - cropped_right - cropped_left
            label_ltrb[label_ltrb[:, 3] > 1 - cropped_bottom - cropped_top, 3] = 1 - cropped_bottom - cropped_top
            label_ltrb[label_ltrb < 0] = 0
            label_ltrb[:, [0, 2]] /= 1-cropped_right-cropped_left
            label_ltrb[:, [1, 3]] /= 1 - cropped_bottom - cropped_top
            #converte back
            label[:, 0] = (label_ltrb[:, 0] + label_ltrb[:, 2])/2
            label[:, 1] = (label_ltrb[:, 1] + label_ltrb[:, 3])/2
            label[:, 2] = label_ltrb[:, 2] - label_ltrb[:, 0]
            label[:, 3] = label_ltrb[:, 3] - label_ltrb[:, 1]

            ignore_w = label[:, 2] <= 0
            ignore_h = label[:, 3] <= 0
            ignore = np.logical_or(ignore_w, ignore_h)
            label = label[np.logical_not(ignore)]
            # crop image
            new_xmin = int(cropped_left * width)
            new_ymin = int(cropped_top * height)
            new_xmax = int(width * (1 - cropped_right))
            new_ymax = int(height * (1 - cropped_bottom))
            image = image[new_ymin:new_ymax, new_xmin:new_xmax, :]
        return image, label


class VerticalFlip(object):
    # tested works
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, data):
        image, label = data
        if uniform(0, 1) < self.prob:
            image = cv2.flip(image, 1)
            label[:, 0] = 1 - label[:, 0]
        return image, label


class HSVAdjust(object):
    # tested works
    def __init__(self, hue=30, saturation=1.5, value=1.5, prob=0.5):
        super().__init__()
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.prob = prob

    def clip_hue(self, hue_channel):
        hue_channel[hue_channel >= 360] -= 360
        hue_channel[hue_channel < 0] += 360
        return hue_channel

    def __call__(self, data):
        image, label = data
        if uniform(0, 1) < self.prob:
            adjust_hue = uniform(-self.hue, self.hue)
            adjust_saturation = uniform(1/self.saturation, self.saturation)
            adjust_value = uniform(1/self.value, self.value)
            image = image.astype(np.float32) / 255
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image[:, :, 0] += adjust_hue
            image[:, :, 0] = self.clip_hue(image[:, :, 0])
            image[:, :, 1] = np.clip(adjust_saturation * image[:, :, 1], 0.0, 1.0)
            image[:, :, 2] = np.clip(adjust_value * image[:, :, 2], 0.0, 1.0)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            image = (image * 255).astype(np.uint8)
        return image, label


class Resize(object):
    # todo implement random size(x*32) since fully convolutional
    # todo normalized or not normalized box coordinates ??? what does the loss need
    # todo all need to be also working for sequenze
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def __call__(self, data):
        image, label = data
        image = cv2.resize(image, (self.image_size, self.image_size))
        width = height = self.image_size
        #label[:, [0, 2]] *= width
        #label[:, [1, 3]] *= height
        return image, label
