
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .mlp_gat import Denoising_Unet
import numpy as np
from .MS_SSIM_loss import compute_ms_ssim_loss
from .filter_layer import AGFNN, AGFNN_GAT, AGFNN_GIN
from .positional_encoding import Posintion_learner


def extract(v, t, x_shape):
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class SDMG(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            norm: Optional[str],
            alpha_l: float = 2,
            beta_schedule: str = 'linear',
            beta_1: float = 0.0001,
            beta_T: float = 0.02,
            T: int = 1000,
            weights=[1, 0.5, 0.3],
            filter_num_layers=2,
            filter_nhead=2,
            filter_attn_drop=0.3,
            filter_feat_drop=0.1,
            num_pos_layers=2,
            RW_step=8,
            **kwargs

         ):
        super(SDMG, self).__init__()
        self.T = T
        self.weights = weights
        beta = get_beta_schedule(beta_schedule, beta_1, beta_T, T)
        self.register_buffer(
                'betas', beta
                )
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
                'sqrt_alphas_bar', torch.sqrt(alphas_bar)
                )
        self.register_buffer(
                'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar)
                )

        self.alpha_l = alpha_l
        assert num_hidden % nhead == 0
        self.net = Denoising_Unet(in_dim=in_dim,
                                  num_hidden=num_hidden,
                                  out_dim=in_dim,
                                  num_layers=num_layers,
                                  nhead=nhead,
                                  activation=activation,
                                  feat_drop=feat_drop,
                                  attn_drop=attn_drop,
                                  negative_slope=0.2,
                                  norm=norm)

        self.time_embedding = nn.Embedding(T, num_hidden)
        self.filter_layer = AGFNN_GAT(in_dim, hidden_size=num_hidden, out_size=num_hidden, head=filter_nhead, feat_drop=filter_feat_drop, attn_drop=filter_attn_drop, negative_slope=0.2, num_layers=filter_num_layers)

        ksteps = list(range(1, RW_step))
        self.num_pos_layers = num_pos_layers
        self.positional_encoder = nn.ModuleList()
        self.positional_encoder.append(
            Posintion_learner(len(ksteps), hidden_dim=num_hidden, out_dim=num_hidden, norm='layernorm', activation='prelu'))
        for l in range(self.num_pos_layers-1):
            self.positional_encoder.append(Posintion_learner(num_hidden, hidden_dim=num_hidden, out_dim=num_hidden, norm='layernorm', activation='prelu'))

    def forward(self, g, x):
        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1], ))

        t = torch.randint(self.T, size=(x.shape[0], ), device=x.device)
        x_t, time_embed, g, filter_embed, position_embed = self.sample_q(t, x, g)

        loss = self.node_denoising(x, x_t, time_embed, g, filter_embed, position_embed)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def sample_q(self, t, x, g):
        miu, std = x.mean(dim=0), x.std(dim=0)
        noise = torch.randn_like(x, device=x.device)
        with torch.no_grad():
            noise = F.layer_norm(noise, (noise.shape[-1], ))
        noise = noise * std + miu
        noise = torch.sign(x) * torch.abs(noise)
        x_t = (
                extract(self.sqrt_alphas_bar, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_bar, t, x.shape) * noise
                )
        time_embed = self.time_embedding(t)
        filter_embed = self.filter_layer(g, x)
        adj_pos = g.ndata["adj_pos"]
        for l in range(self.num_pos_layers):
            position_embed = self.positional_encoder[l](adj_pos)
            adj_pos = position_embed
        return x_t, time_embed, g, filter_embed, position_embed

    def node_denoising(self, x, x_t, time_embed, g, filter_embed, position_embed):
        out, _ = self.net(g, x_t=x_t, time_embed=time_embed, filter_embed=filter_embed, position_embed=position_embed)
        loss = loss_fn(g, out, x, self.weights, self.alpha_l)

        return loss

    def embed(self, g, x, T):
        t = torch.full((1, ), T, device=x.device)
        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1], ))
        x_t, time_embed, g, filter_embed, position_embed = self.sample_q(t, x, g)
        _, hidden = self.net(g, x_t=x_t, time_embed=time_embed, filter_embed=filter_embed, position_embed=position_embed)
        return hidden


def loss_fn(graph, x, y, weights, alpha=2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    scales = list(range(1, len(weights)))
    loss = compute_ms_ssim_loss(graph, x, y, scales, weights)
    return loss


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas)
