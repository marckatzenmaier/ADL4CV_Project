
import torch.nn as nn
import torch
from yolo.yolo_encoder import YoloEncoder
from yolo.yolo_lstm_part import YoloLSTM_part


class YoloLSTM(nn.Module):
    def __init__(self, batch_size, freeze_encoder=True,
                 anchors=[(0.215, 0.8575), (0.3728125, 1.8225), (0.621875, 2.96625),
                          (1.25, 6.12), (3.06125, 11.206875)]):
        super(YoloLSTM, self).__init__()
        self.anchors = anchors
        self.batch_size = batch_size
        self.encoder = YoloEncoder(self.batch_size)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.lstm_part = YoloLSTM_part(self.batch_size)

    def forward(self, input):

        output = self.encoder.forward(input)
        output = self.lstm_part.forward(output)
        return output

    def reinit_lstm(self):
        self.lstm_part.reinit_lstm()

