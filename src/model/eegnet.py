import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class EEGNet(nn.Module):
    def __init__(self, args, num_classes, chans=62, samples=400, dropout_rate=0.5, kern_length=200, F1=8, D=2, F2=16,
                 norm_rate=0.25):
        super(EEGNet, self).__init__()

        self.num_classes = num_classes
        self.chans = chans
        self.samples = samples
        self.dropout_rate = dropout_rate
        self.kern_length = kern_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate

        # First Conv2D Layer
        self.conv1 = nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise Conv2D Layer
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Separable Conv2D Layer
        self.separable_conv1 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.separable_conv2 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 10))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Fully Connected Layer
        self.fc1 = nn.Linear(F2 * (samples // 40), num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.separable_conv1(x)
        x = self.separable_conv2(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.flatten(start_dim=1)
        x = self.fc1(x)

        return x
