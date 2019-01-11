
import torch.nn as nn
import torch
from lstm.LSTMModels import ConvLSTMCell_true
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class YoloLSTM_part(nn.Module):
    def __init__(self, batch_size, anchors=[(0.215, 0.8575), (0.3728125, 1.8225), (0.621875, 2.96625),
                                            (1.25, 6.12), (3.06125, 11.206875)]):
        super(YoloLSTM_part, self).__init__()
        self.anchors = anchors

        self.lstm_cell = ConvLSTMCell_true(input_size=(13, 13), input_dim=1024+256, hidden_dim=1024,
                                                            kernel_size=(3, 3), bias=True)
        self.reinit_lstm(batch_size)
        self.stage3_conv2 = nn.Conv2d(1024, len(self.anchors) * 5, 1, 1, 0, bias=False)

    def forward(self, input):
        self.hidden, self.cell = self.lstm_cell(input, (self.hidden, self.cell))
        output = self.hidden
        output = self.stage3_conv2(output)
        return output

    def reinit_lstm(self, batch_size):
        dev = next(self.parameters()).device  # torch.device("cuda")#self.hidden.device
        self.hidden, self.cell = (Variable(torch.zeros(batch_size, self.lstm_cell.hidden_dim,
                                                        self.lstm_cell.height, self.lstm_cell.width).to(dev)),
                                  Variable(torch.zeros(batch_size, self.lstm_cell.hidden_dim,
                                                        self.lstm_cell.height, self.lstm_cell.width).to(dev)))