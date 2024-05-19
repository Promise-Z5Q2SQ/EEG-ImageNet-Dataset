import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, args, num_classes, chans=62, dropout_rate=0.5):
        super(MLP, self).__init__()

        self.num_classes = num_classes
        self.chans = chans
        input_dim = chans * 5

        self.l1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.l2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.l3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.bn1(self.l1(x))))
        x = self.dropout(F.relu(self.bn2(self.l2(x))))
        x = self.l3(x)
        return x
