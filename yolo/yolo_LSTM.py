"""
@author: Marc Katzenmaier
@author: Nikita Kister
"""
import torch.nn as nn
from yolo.yolo_encoder import YoloEncoder
from Flow import FlowNetSEncoder
from yolo.yolo_lstm_part import YoloLSTM_part
import torchvision.transforms as transforms
from yolo.yolo_utils import *


class YoloLSTM(nn.Module):
    """
    our YoloLSTM model based on the yolo encoder and convolutional LSTM(yolo_lstm_part)
    """
    def __init__(self, batch_size, image_size=416, freeze_encoder=True,
                 anchors=[(0.215, 0.8575), (0.3728125, 1.8225), (0.621875, 2.96625),
                          (1.25, 6.12), (3.06125, 11.206875)]):
        super(YoloLSTM, self).__init__()
        self.image_size = image_size
        self.anchors = anchors
        self.encoder = YoloEncoder()
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.lstm_part = YoloLSTM_part(batch_size=batch_size, input_size=(image_size//32, image_size//32))

    def forward(self, input):

        output = self.encoder.forward(input)
        output = self.lstm_part.forward(output)
        return output

    def reinit_lstm(self, batch_size):
        """
        used to reinit the hidden state of the LSTM should be called always if the batch_size changes,
         otherwise there will occur error
        :param batch_size: input batch size need to be known and can't change
        """
        self.lstm_part.reinit_lstm(batch_size)

    def predict_modified_logits(self, img):
        """returns all box parameters and their confidence"""
        self.eval()
        trans = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()])
        sample = trans(img).unsqueeze(0)
        sample = Variable(torch.FloatTensor(sample))
        device = next(self.parameters()).device
        if torch.cuda.is_available() and device.type == "cuda":
            sample = sample.cuda()
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

    def load_pretrained_weights(self, path):
        """
        loads the encoder path of a given snapshot
        :param path: path of an Yolo snapshot
        """
        device = next(self.encoder.parameters()).device
        load_strict = False
        model_state_dict = torch.load(path, map_location=device)['model_state_dict']
        del model_state_dict["stage3_conv2.weight"]
        del model_state_dict["stage3_conv1.0.weight"]
        del model_state_dict["stage3_conv1.1.weight"]
        del model_state_dict["stage3_conv1.1.bias"]
        del model_state_dict["stage3_conv1.1.running_mean"]
        del model_state_dict["stage3_conv1.1.running_var"]
        del model_state_dict["stage3_conv1.1.num_batches_tracked"]
        self.encoder.load_state_dict(model_state_dict, strict=load_strict)


class YoloFlowLSTM(nn.Module):
    def __init__(self, batch_size, image_size=416, freeze_encoder=True,
                 anchors=[(0.215, 0.8575), (0.3728125, 1.8225), (0.621875, 2.96625),
                          (1.25, 6.12), (3.06125, 11.206875)]):
        super(YoloFlowLSTM, self).__init__()
        self.image_size = image_size
        self.anchors = anchors
        self.encoder = YoloEncoder()
        self.flownet = FlowNetSEncoder(False)

        # reduce flow dimensions to 13 by 13 with 512 channels
        self.conv_flow = nn.Conv2d(512 + 2, 512, kernel_size=2, bias=True)  # not sure about bias
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.flownet.parameters():
                param.requires_grad = False
        self.lstm_part = YoloLSTM_part(input_dim=1024+512, batch_size=batch_size)  # todo set the right dimension

    def forward(self, input):
        yolo_input, flow_input = input
        yolo_encoding = self.encoder.forward(yolo_input)
        flow_encoding = self.flownet(flow_input)
        flow_encoding = self.conv_flow(flow_encoding)

        encoding = torch.cat((yolo_encoding, flow_encoding), 1)
        output = self.lstm_part.forward(encoding)
        return output

    def reinit_lstm(self, batch_size):
        self.lstm_part.reinit_lstm(batch_size)





