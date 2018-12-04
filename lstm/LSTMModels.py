import torch.nn as nn
import torch
import numpy as np


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
