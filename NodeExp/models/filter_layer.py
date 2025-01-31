import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import ChebConv
from dgl.nn.pytorch import GATConv


class AGFNN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, K, num_layers, activation="relu"):
        super(AGFNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = create_activation(activation)
        self.layers.append(ChebConv(in_feats, hidden_size, K))
        for i in range(num_layers - 2):
            self.layers.append(ChebConv(hidden_size, hidden_size, K))
        self.layers.append(ChebConv(hidden_size, out_feats, K))

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
        return h

class AGFNN_GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, out_size, head, feat_drop, attn_drop, negative_slope, num_layers, activation="relu"):
        super(AGFNN_GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = create_activation(activation)
        self.layers.append(GATConv(in_feats, hidden_size // head, head, feat_drop, attn_drop, negative_slope))
        for i in range(num_layers - 2):
            self.layers.append(GATConv(hidden_size, hidden_size // head, head, feat_drop, attn_drop, negative_slope))
        self.layers.append(GATConv(hidden_size, out_size, 1, feat_drop, attn_drop, negative_slope))

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            h = h.flatten(1)
            if i != len(self.layers) - 1:
                h = self.activation(h)
        return h

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")