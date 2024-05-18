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
        self.l2 = nn.Linear(12000, 6000)
        self.l3 = nn.Linear(6000, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.l1(x)))
        x = self.dropout(F.relu(self.l2(x)))
        x = self.l3(x)
        return x
