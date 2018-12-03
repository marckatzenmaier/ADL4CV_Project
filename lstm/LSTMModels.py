import torch.nn as nn
import torch
import numpy as np


class SequenceClassifier(nn.Module):

    def __init__(self, input_size, output_size, batch_size_init):
        super(SequenceClassifier, self).__init__()
        # layer parameter
        self.lstm_input_size = 128
        self.hidden_dim = 128
        # todo 2-3 conv layer
        # lstm cell
        self.feature_transform = nn.Linear(np.prod(input_size), self.lstm_input_size)
        self.lstm = nn.LSTMCell(self.lstm_input_size, self.hidden_dim)
        self.output_transform = nn.Linear(self.hidden_dim, np.prod(output_size))

        self.relu = nn.ReLU()
        self.hidden = torch.zeros(batch_size_init, self.hidden_dim)
        self.cell = torch.zeros(batch_size_init, self.hidden_dim)
        # idea use convolutions for the input and then

    def forward(self, input):
        batch_size = input.shape[0]
        x = self.feature_transform(input.view(batch_size, -1))
        x = self.relu(x)
        self.hidden, self.cell = self.lstm(x.view(batch_size, -1), (self.hidden, self.cell))
        return self.output_transform(self.hidden.view(batch_size, -1)).view(batch_size, 16, 16, 5, 4)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)
