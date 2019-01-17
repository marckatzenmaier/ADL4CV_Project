import torch.nn as nn
import torch as torch
import numpy as np
from torch.autograd import Variable


class SequenceClassifier(nn.Module):

    def __init__(self, input_size, output_size, batch_size_init):
        super(SequenceClassifier, self).__init__()
        # layer parameter

        self.lstm_input_size = 512
        self.hidden_dim = 512
        self.init_feature_transform(input_size)
        # todo 2-3 conv layer

        # lstm cell
        self.lstm = nn.LSTMCell(self.lstm_input_size, self.hidden_dim)

        self.init_output_transform(output_size)

        self.relu = nn.ReLU()
        self.hidden = torch.zeros(batch_size_init, self.hidden_dim)
        self.cell = torch.zeros(batch_size_init, self.hidden_dim)
        # idea use convolutions for the input and then

    def forward(self, input):
        batch_size = input.shape[0]
        x = self.feature_transform(input.view(batch_size, -1))
        x = self.relu(x)
        self.hidden, self.cell = self.lstm(x.view(batch_size, -1), (self.hidden, self.cell))
        return self.output_transform(self.hidden.view(batch_size, -1))

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)

    def reset_hidden(self, batch_size):
        self.hidden, self.cell = self.init_hidden(batch_size), self.init_hidden(batch_size)

    def feature_transform(self, x):
        batch_size = x.size(0)
        x = self.relu(self.feature_transform_1(x.view(batch_size, -1)))
        x = self.relu(self.feature_transform_2(x.view(batch_size, -1)))
        x = self.relu(self.feature_transform_3(x.view(batch_size, -1)))

        return x

    def output_transform(self, x):
        batch_size = x.size(0)
        return self.output_transform_1(x).view(batch_size, 16, 16, 5, 4)

    def init_feature_transform(self, input_size):
        self.feature_transform_1 = nn.Linear(np.prod(input_size), self.lstm_input_size)
        self.feature_transform_2 = nn.Linear(self.lstm_input_size, self.lstm_input_size)
        self.feature_transform_3 = nn.Linear(self.lstm_input_size, self.lstm_input_size)

    def init_output_transform(self, output_size):
        self.output_transform_1 = nn.Linear(self.hidden_dim, np.prod(output_size))


class SequenceClassifierCNN(nn.Module):

    def __init__(self, input_size, output_size, batch_size_init):
        super(SequenceClassifierCNN, self).__init__()
        # layer parameter

        #self.lstm_input_size = 126*16*16
        #self.hidden_dim = 512
        self.init_feature_transform(input_size)

        # lstm cell
        #self.lstm_1 = ConvLSTMCell(channels_in=64, hidden_size=(64, 16, 16), kernel=(3, 3))
        #self.lstm_2 = ConvLSTMCell(channels_in=64, hidden_size=(64, 16, 16), kernel=(3, 3))

        self.lstm_1 = ConvLSTMCell((16, 16), 64, 128, (1, 1), bias=True)

        self.init_output_transform(output_size)

        self.relu = nn.ReLU()
        self.hidden_1 = torch.zeros([batch_size_init, 128, 16, 16])
        self.cell_1 = torch.zeros([batch_size_init, 128, 16, 16])

        #self.hidden_2 = torch.zeros([batch_size_init, 64, 16, 16])
        #self.cell_2 = torch.zeros([batch_size_init, 64, 16, 16])
        # idea use convolutions for the input and then

    def forward(self, input):
        batch_size = input.shape[0]
        input_h = input.shape[1]
        input_w = input.shape[2]
        x = self.feature_transform(input.unsqueeze(1).view(batch_size, 1, input_h, input_w, -1).permute(0, 1, 4, 2, 3))

        self.hidden_1, self.cell_1 = self.lstm_1(x.squeeze(dim=2), (self.hidden_1, self.cell_1))

        #self.hidden_2, self.cell_2 = self.lstm_2(self.hidden_1, (self.hidden_2, self.cell_2))
        return self.output_transform(self.hidden_1)

    def reset_hidden(self, batch_size):
        self.hidden_1, self.cell_1 = self.lstm_1.init_hidden(batch_size)
        #self.hidden_1 = torch.autograd.Variable(torch.zeros([batch_size, 64, 16, 16]))
        #self.hidden_2 = torch.autograd.Variable(torch.zeros([batch_size, 64, 16, 16]))
        #self.cell_1 = torch.autograd.Variable(torch.zeros([batch_size, 64, 16, 16]))
        #self.cell_2 = torch.autograd.Variable(torch.zeros([batch_size, 64, 16, 16]))

    def feature_transform(self, x):
        x = self.relu(self.conv_1(x))
        return x

    def output_transform(self, x):
        x = self.conv_1_out(x)
        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]
        return x.permute(0, 2, 3, 1).view(batch_size, height, width, 5, -1)

    def init_feature_transform(self, input_size):
        self.conv_1 = nn.Conv3d(1, 64, (input_size[3] * input_size[2], 1, 1), stride=(input_size[2], 1, 1))

    def init_output_transform(self, output_size):
        self.conv_1_out = nn.Conv2d(128, 5, (1, 1))


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        dev = next(self.parameters()).device
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(dev)),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(dev)))


class ConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, batch_size_init):
        super(ConvLSTM, self).__init__()
        # layer parameter

        self.input = nn.Conv2d(input_dim, 512, kernel_size=3, padding=1)

        self.lstm = ConvLSTMCell(input_size, 512, hidden_dim, kernel_size=(3, 3), bias=True)

        self.output = nn.Conv2d(hidden_dim, input_dim, kernel_size=3, padding=1)

        self.hidden, self.cell = self.lstm.init_hidden(batch_size_init)

    def forward(self, input):

        x = self.input(input)  # todo bigger embedding

        self.hidden, self.cell = self.lstm(x.squeeze(dim=2), (self.hidden, self.cell))

        return self.output(self.hidden)

    def reset_hidden(self, batch_size):
        self.hidden, self.cell = self.lstm.init_hidden(batch_size)

    def set_hidden(self, hidden, cell):
        self.hidden = hidden
        self.cell = cell
