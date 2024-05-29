import mne
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import global_add_pool, SGConv
from torch_scatter import scatter_add

channels = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5",
            "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
            "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2",
            "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]


def get_edge_weight():
    montage = mne.channels.read_dig_fif('../data/mode/' + 'montage.fif')
    montage.ch_names = json.load(open('../data/mode/' + "montage_ch_names.json"))
    edge_pos = montage.get_positions()['ch_pos']
    edge_weight = np.zeros([len(channels), len(channels)])
    edge_pos_value = [edge_pos[key] for key in channels]
    delta = 4710000000
    edge_index = [[], []]
    for i in range(len(channels)):
        for j in range(len(channels)):
            edge_index[0].append(i)
            edge_index[1].append(j)
            if i == j:
                edge_weight[i][j] = 1
            else:
                edge_weight[i][j] = np.sum([(edge_pos_value[i][k] - edge_pos_value[j][k]) ** 2 for k in range(3)])
                edge_weight[i][j] = min(1, delta / edge_weight[i][j])
    global_connections = [['FP1', 'FP2'], ['AF3', 'AF4'], ['F5', 'F6'], ['FC5', 'FC6'], ['C5', 'C6'],
                          ['CP5', 'CP6'], ['P5', 'P6'], ['PO5', 'PO6'], ['O1', 'O2']]
    for item in global_connections:
        i, j = item
        if i in channels and j in channels:
            i = channels.index(item[0])
            j = channels.index(item[1])
            edge_weight[i][j] -= 1
            edge_weight[j][i] -= 1
    return edge_index, edge_weight


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    # reposition the diagonal values to the end
    # actually return num_nodes
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index
    mask = row != col
    inv_mask = ~mask  # diagonal positions
    loop_weight = torch.full(
        (num_nodes,),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
        device=edge_index.device)
    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        remaining_edge_weight = edge_weight[inv_mask]
        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)
    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)
    return edge_index, edge_weight


class NewSGConv(SGConv):
    def __init__(self, num_features, num_classes, K=1, cached=False, bias=True):
        super(NewSGConv, self).__init__(num_features, num_classes, K=K, cached=cached, bias=bias)
        torch.nn.init.xavier_normal_(self.lin.weight)

    # allow negative edge weights
    @staticmethod
    # Note: here, num_nodes = self.num_nodes * batch_size
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(torch.abs(edge_weight), row, dim=0,
                          dim_size=num_nodes)  # calculate degreematrix, i.e, D(stretched) in the paper.
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # calculate normalized adjacency matrix, i.e, S(stretched) in the paper.
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        if not self.cached or self.cached_result is None:
            edge_index, norm = NewSGConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype, )
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x
        return self.lin(self.cached_result)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RGNN(torch.nn.Module):
    def __init__(self, device, num_nodes, edge_weight, edge_index, num_features, num_hiddens, num_classes, num_layers,
                 dropout=0.5, domain_adaptation=False):
        """
            num_nodes: number of nodes in the graph
            learn_edge_weight: if True, the edge_weight is learnable
            edge_weight: initial edge matrix
            num_features: feature dim for each node/channel
            num_hiddens: a tuple of hidden dimensions
            num_classes: number of emotion classes
            num_layers: number of layers
            dropout: dropout rate in final linear layer
            domain_adaptation: RevGrad
        """
        learn_edge_weight = True
        super(RGNN, self).__init__()
        self.device = device
        self.domain_adaptation = domain_adaptation
        self.num_nodes = num_nodes
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        self.edge_index = torch.tensor(edge_index)
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[
            self.xs, self.ys]  # strict lower triangular values
        self.edge_weight = nn.Parameter(torch.Tensor(edge_weight).float(), requires_grad=learn_edge_weight)
        self.dropout = dropout
        self.conv1 = NewSGConv(num_features=num_features, num_classes=num_hiddens, K=num_layers)
        self.fc = nn.Linear(num_hiddens, num_classes)
        # xavier init
        torch.nn.init.xavier_normal_(self.fc.weight)
        if self.domain_adaptation == True:
            self.domain_classifier = nn.Linear(num_hiddens, 2)
            torch.nn.init.xavier_normal_(self.domain_classifier.weight)
        self.bn = nn.BatchNorm1d(num_nodes * num_features)
        self.bn1 = nn.BatchNorm1d(num_hiddens)

    def append(self, edge_index, batch_size):  # stretch and repeat and rename
        edge_index_all = torch.LongTensor(2, edge_index.shape[1] * batch_size)
        data_batch = torch.LongTensor(self.num_nodes * batch_size)
        for i in range((batch_size)):
            edge_index_all[:, i * edge_index.shape[1]:(i + 1) * edge_index.shape[1]] = edge_index + i * self.num_nodes
            data_batch[i * self.num_nodes:(i + 1) * self.num_nodes] = i
        return edge_index_all.to(self.device), data_batch.to(self.device)

    def forward(self, x, alpha=0, need_pred=True, need_dat=False):
        batch_size = len(x)
        x = self.bn(x)  # (80, 310)
        x = x.view(-1, 5)  # (4960, 5)
        edge_index, data_batch = self.append(self.edge_index, batch_size)
        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1, 0) - torch.diag(edge_weight.diagonal())
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        # edge_index: (2, self.num_nodes * self.num_nodes * batch_size)
        # edge_weight: (self.num_nodes * self.num_nodes * batch_size,)
        x = F.relu(self.conv1(x, edge_index, edge_weight))  # (4960, 200)
        # domain classification
        domain_output = None
        if need_dat == True:
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x)
        if need_pred == True:
            x = global_add_pool(x, data_batch, size=batch_size)  # (80, 200)
            x = self.bn1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc(x)
        # x.shape -> (batch_size, num_classes)
        # domain_output.shape -> (batch_size * num_nodes, 2)
        if domain_output is not None:
            return x, domain_output
        else:
            return x
