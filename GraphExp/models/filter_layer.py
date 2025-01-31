import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import ChebConv
from dgl.nn.pytorch import GATConv, GINConv
from .mlp_gat import MlpBlock


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


class AGFNN_GIN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_size, num_layers, activation="relu", norm='layernorm'):

        super(AGFNN_GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = create_activation(activation)

        def build_mlp(input_dim, output_dim):
            return MlpBlock(
                in_dim=input_dim,
                hidden_dim=hidden_size,
                out_dim=output_dim,
                norm=norm,
                activation=activation
            )

        self.layers.append(GINConv(build_mlp(in_feats, hidden_size)))

        for _ in range(num_layers - 2):
            self.layers.append(GINConv(build_mlp(hidden_size, hidden_size)))

        self.layers.append(GINConv(build_mlp(hidden_size, out_size)))

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
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