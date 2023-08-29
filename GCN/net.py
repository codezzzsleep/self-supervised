import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MyNet(nn.Module):
    def __init__(self, num_feature, num_hidden, num_classes, dropout):
        super(MyNet, self).__init__()
        self.gc1 = GCNConv(num_feature, num_hidden)
        self.gc2 = GCNConv(num_hidden, num_classes)
        self.activate = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.activate(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)
