import os

import torch.nn as nn
import torch.nn.functional as F


class MyCNN(nn.Module):

    def __init__(self, INPUT_SIZE):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=3, padding=0)

        # Batch normalization layers
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(self.afterConvs(INPUT_SIZE) * self.afterConvs(INPUT_SIZE) * 64, 1024)
        self.fc2 = nn.Linear(1024, 30)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.normal_(self.fc2.weight, std=0.01)

    def forward(self, x):
        # Apply convolutional layers, batch normalization, and ReLU activation
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))

        # Flatten the tensor for the fully connected layer
        x = x.flatten(start_dim=1)
        # Apply fully connected layers with Sigmoid and dropout
        x = F.sigmoid(self.fc1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))

        return x

    def conv_output_size(self, input_size, filter_size, stride, padding):
        output_size = ((input_size - filter_size + 2 * padding) // stride) + 1

        return output_size

    def afterConvs(self, input_size):
        input_size = self.conv_output_size(input_size, 3, 1, 0)
        input_size = self.conv_output_size(input_size, 2, 2, 0)
        input_size = self.conv_output_size(input_size, 3, 3, 0)
        input_size = self.conv_output_size(input_size, 2, 2, 0)
        return input_size