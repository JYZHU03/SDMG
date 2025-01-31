
import torch
import torch.nn as nn
from functools import partial
from torch_geometric.utils import (to_dense_adj, scatter)
from torch_geometric.utils.num_nodes import maybe_num_nodes

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


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity

def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)

    return rw_landing

class Residual(nn.Module):
    def __init__(self, fnc):
        super().__init__()
        self.fnc = fnc

    def forward(self, x, *args, **kwargs):
        return self.fnc(x, *args, **kwargs) + x

class Posintion_learner(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 norm: str = 'layernorm', activation: str = 'prelu'):
        super(Posintion_learner, self).__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.res_mlp = Residual(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            create_norm(norm)(hidden_dim),
            create_activation(activation),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.act = create_activation(activation)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.res_mlp(x)
        x = self.out_proj(x)
        x = self.act(x)
        return x

