import torch
from torch import nn

EPS = 1e-6


class NormalizationLayer(nn.Module):
    def __init__(self, algebra, features, init: float = 0):
        super().__init__()
        self.algebra = algebra
        self.in_features = features

        self.a = nn.Parameter(torch.zeros(self.in_features, algebra.n_subspaces) + init)

    def forward(self, input):
        assert input.shape[1] == self.in_features

        norms = torch.cat(self.algebra.norms(input), dim=-1)
        s_a = torch.sigmoid(self.a)
        norms = s_a * (norms - 1) + 1  # Interpolates between 1 and the norm.
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        normalized = input / (norms + EPS)

        return normalized
