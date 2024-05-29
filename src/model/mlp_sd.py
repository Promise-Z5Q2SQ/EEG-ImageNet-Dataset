import torch.nn as nn


class MLPMapper(nn.Module):
    def __init__(self, input_dim=310, output_dim=77 * 768, dropout_rate=0.5):
        super(MLPMapper, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dropout(self.relu(x))
        x = self.fc3(x)
        x = x.view(x.size(0), 77, 768)
        return x
