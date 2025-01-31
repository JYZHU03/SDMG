import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion(nn.Module):
    def __init__(self, embed_dim, num_embeds):
        super(GatedFusion, self).__init__()
        self.gate = nn.Linear(embed_dim * num_embeds, embed_dim)

    def forward(self, *embeds):
        combined = torch.cat(embeds, dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        fused = gate * embeds[0]
        for embed in embeds[1:]:
            fused += gate * embed
        return fused
