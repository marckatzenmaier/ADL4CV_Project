"""
@author: Viet Nguyen <nhviet1009@gmail.com>
@author: Marc Katzenmaier (added the helper functions)
"""
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from yolo.yolo_utils import *


class Yolo(nn.Module):
    def __init__(self, image_size=416, anchors=[(0.215, 0.8575), (0.3728125, 1.8225), (0.621875, 2.96625),
                                            (1.25, 6.12), (3.06125, 11.206875)]):
        super(Yolo, self).__init__()
        self.anchors = anchors
        self.image_size = image_size

        self.stage1_conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv6 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv7 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv8 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv9 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv10 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv11 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv12 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv13 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))

        self.stage2_a_maxpl = nn.MaxPool2d(2, 2)
        self.stage2_a_conv1 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(1024), nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv2 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False), nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv3 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv4 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False), nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv5 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv6 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv7 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))

        self.stage2_b_conv = nn.Sequential(nn.Conv2d(512, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                           nn.LeakyReLU(0.1, inplace=True))

        self.stage3_conv1 = nn.Sequential(nn.Conv2d(256 + 1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage3_conv2 = nn.Conv2d(1024, len(self.anchors) * 5, 1, 1, 0, bias=False)

    def forward(self, input):
        output = self.stage1_conv1(input)
        output = self.stage1_conv2(output)
        output = self.stage1_conv3(output)
        output = self.stage1_conv4(output)
        output = self.stage1_conv5(output)
        output = self.stage1_conv6(output)
        output = self.stage1_conv7(output)
        output = self.stage1_conv8(output)
        output = self.stage1_conv9(output)
        output = self.stage1_conv10(output)
        output = self.stage1_conv11(output)
        output = self.stage1_conv12(output)
        output = self.stage1_conv13(output)

        residual = output

        output_1 = self.stage2_a_maxpl(output)
        output_1 = self.stage2_a_conv1(output_1)
        output_1 = self.stage2_a_conv2(output_1)
        output_1 = self.stage2_a_conv3(output_1)
        output_1 = self.stage2_a_conv4(output_1)
        output_1 = self.stage2_a_conv5(output_1)
        output_1 = self.stage2_a_conv6(output_1)
        output_1 = self.stage2_a_conv7(output_1)

        output_2 = self.stage2_b_conv(residual)
        batch_size, num_channel, height, width = output_2.data.size()
        output_2 = output_2.view(batch_size, int(num_channel / 4), height, 2, width, 2).contiguous()
        output_2 = output_2.permute(0, 3, 5, 1, 2, 4).contiguous()
        output_2 = output_2.view(batch_size, -1, int(height / 2), int(width / 2))

        output = torch.cat((output_1, output_2), 1)
        output = self.stage3_conv1(output)
        output = self.stage3_conv2(output)

        return output

    def predict_modified_logits(self, img):
        """returns all box parameters and their confidence"""
        self.eval()
        trans = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()])
        sample = trans(img).unsqueeze(0)
        sample = Variable(torch.FloatTensor(sample))
        device = next(self.parameters()).device
        sample = sample.to(device)
        with torch.no_grad():
            logits = self(sample)
            modified_logits = logits_to_box_params(logits, self.anchors)
        return modified_logits

    def predict_boxes(self, img, nms_threshold=0.5, conf_threshold=0.25):
        """
        Runs the network itself and the postprocessing
        """
        box_logits = self.predict_modified_logits(img)
        predictions = filter_box_params(box_logits, self.image_size, self.anchors, conf_threshold, nms_threshold)
        return predictions

    def load_snapshot(self, path):
        """
        loads a snapshot of this model
        :param path: snapshot path
        """
        device = next(self.parameters()).device
        model_state_dict = torch.load(path, map_location=device)['model_state_dict']
        self.load_state_dict(model_state_dict, strict=False)

    def load_pretrained_weights(self, path, reinitailise_last_layer=True):
        """
        load the given pretrained weights
        :param path: path to the weights
        :param reinitailise_last_layer: reinitialise the last layer if True
        :return:
        """
        load_strict = False
        device = next(self.parameters()).device
        model_state_dict = torch.load(path, map_location=device)
        if reinitailise_last_layer:
            del model_state_dict["stage3_conv2.weight"]
        self.load_state_dict(model_state_dict, strict=load_strict)
        self.to(device)
        if reinitailise_last_layer:
            nn.init.normal_(list(self.modules())[-1].weight, 0, 0.01)



