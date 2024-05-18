import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, args, num_classes, chans=62, samples=400, dropout_rate=0.5):
        super(MLP, self).__init__()

        self.num_classes = num_classes
        self.chans = chans
        self.samples = samples
        input_dim = chans * samples

        self.l1 = nn.Linear(input_dim, 12000)
        self.bn1 = nn.BatchNorm1d(12000)
        self.l2 = nn.Linear(12000, 6000)
        self.bn2 = nn.BatchNorm1d(6000)
        self.l3 = nn.Linear(6000, 3000)
        self.bn3 = nn.BatchNorm1d(3000)
        self.l4 = nn.Linear(3000, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.l5 = nn.Linear(1000, 200)
        self.bn5 = nn.BatchNorm1d(200)
        self.l6 = nn.Linear(200, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.bn1(self.l1(x))))
        x = self.dropout(F.relu(self.bn2(self.l2(x))))
        x = self.dropout(F.relu(self.bn3(self.l3(x))))
        x = self.dropout(F.relu(self.bn4(self.l4(x))))
        x = self.dropout(F.relu(self.bn5(self.l5(x))))
        x = self.l6(x)
        return x
