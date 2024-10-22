"""
@author Nikita Kister
taken from https://github.com/ClementPinard/FlowNetPytorch and adapted
"""
import torch
import torch.nn as nn
from numpy import roll


def preprocess(image):
    """
    flownet expects bgr images in range (-.5 .5)
    :param image: images should be a numpy array with shape (batch_size, channel, height, width)
    :return: images ready for flownet in bgr and (-0.5, 0.5) in same shape as input
    """
    # expects torch tensor !!!
    image = image.numpy().transpose((0, 2, 3, 1))
    image = roll(image, 1, axis=-1).transpose((0, 3, 1, 2))  # todo dont know if this roll stays in the batch
    image += -0.5

    return torch.Tensor(image)


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False)
    )


class FlowNetS(nn.Module):
    expansion = 1

    def __init__(self, batch_norm=True):
        super(FlowNetS, self).__init__()

        self.batchNorm = batch_norm
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=False)
#

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.leaky_relu(self.deconv5(out_conv6))

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.leaky_relu(self.deconv4(concat5))

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.leaky_relu(self.deconv3(concat4))

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.leaky_relu(self.deconv2(concat3))

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        return flow2


class FlowNetSEncoder(nn.Module):

    def __init__(self, batch_norm=False, image_size=416):
        """
        truncated FlowNetS for feature extraction.
        :param batch_norm:
        :param image_size: not used. is here to not break code in notebook
        """
        super(FlowNetSEncoder, self).__init__()

        self.batchNorm = batch_norm
        self.conv1   = conv(self.batchNorm,    6,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,   64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm,  128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm,  256,  256)
        self.conv4   = conv(self.batchNorm,  256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm,  512,  512)
        self.conv5   = conv(self.batchNorm,  512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm,  512,  512)
        self.conv6   = conv(self.batchNorm,  512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)

        self.predict_flow6 = predict_flow(1024)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=False)
#

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.leaky_relu(self.deconv5(out_conv6))

        if x.shape[-1] == 416:
            concat5 = torch.cat((out_deconv5, flow6_up), 1)  # first try without tensor from earlier layer
        elif x.shape[-1] == 832:
            concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        else:
            concat5 = None

        return concat5
