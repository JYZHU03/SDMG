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

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_embeds):
        super(AttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_embeds = num_embeds
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, *embeds):

        if len(embeds) != self.num_embeds:
            raise ValueError(f"Expected {self.num_embeds} embeddings, but got {len(embeds)}.")

        stacked_embeds = torch.stack(embeds, dim=1)

        Q = self.query(embeds[0]).unsqueeze(1)

        K = self.key(stacked_embeds)
        V = self.value(stacked_embeds)

        scores = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5))

        fused = torch.matmul(scores, V).squeeze(1)
        return fused

class WeightedFusion(nn.Module):
    def __init__(self, num_embeds):
        super(WeightedFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_embeds))

    def forward(self, *embeds):
        weights = F.softmax(self.weights, dim=0)
        fused = torch.zeros_like(embeds[0])
        for weight, embed in zip(weights, embeds):
            fused += weight * embed
        return fused
