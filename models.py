import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self, in_channels, num_classes, img_height, img_width):
        super(BaseCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        h_dummy = img_height // 4
        w_dummy = img_width // 4
        flattened_dim = 64 * h_dummy * w_dummy

        self.fc1 = nn.Linear(flattened_dim, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DeepCNN(nn.Module):
    def __init__(self, in_channels, num_classes, img_height, img_width):
        super(DeepCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        h_dummy = img_height // 8
        w_dummy = img_width // 8
        flattened_dim = 128 * h_dummy * w_dummy

        self.fc1 = nn.Linear(flattened_dim, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def Method_CNN_MNIST():
    return BaseCNN(in_channels=1, num_classes=10, img_height=28, img_width=28)


def Method_CNN_ORL():
    return BaseCNN(in_channels=1, num_classes=40, img_height=112, img_width=92)


def Method_CNN_CIFAR():
    return BaseCNN(in_channels=3, num_classes=10, img_height=32, img_width=32)


def Method_CNN_CIFAR_Deep():
    return DeepCNN(in_channels=3, num_classes=10, img_height=32, img_width=32)