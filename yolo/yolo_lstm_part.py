
import torch.nn as nn
import torch
from lstm.LSTMModels import ConvLSTMCell
from torch.autograd import Variable


class YoloLSTM_part(nn.Module):
    def __init__(self, input_size=(13, 13), input_dim=1024, hidden_dim=1024, batch_size=1,
                 num_anchors=5):
        super(YoloLSTM_part, self).__init__()

        self.lstm_cell = ConvLSTMCell(input_size=input_size, input_dim=input_dim, hidden_dim=hidden_dim,
                                                            kernel_size=(3, 3), bias=True)
        self.reinit_lstm(batch_size)
        self.stage3_conv2 = nn.Conv2d(1024, num_anchors * 5, 1, 1, 0, bias=False)

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