import torch
from torch import nn

from .utils import unsqueeze_like


class MVSiLU(nn.Module):
    def __init__(self, algebra, channels, invariant="mag2", exclude_dual=False):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.exclude_dual = exclude_dual
        self.invariant = invariant
        self.a = nn.Parameter(torch.ones(1, channels, algebra.dim + 1))
        self.b = nn.Parameter(torch.zeros(1, channels, algebra.dim + 1))

        if invariant == "norm":
            self._get_invariants = self._norms_except_scalar
        elif invariant == "mag2":
            self._get_invariants = self._mag2s_except_scalar
        else:
            raise ValueError(f"Invariant {invariant} not recognized.")

    def _norms_except_scalar(self, input):
        return self.algebra.norms(input, grades=self.algebra.grades[1:])

    def _mag2s_except_scalar(self, input):
        return self.algebra.qs(input, grades=self.algebra.grades[1:])

    def forward(self, input):
        norms = self._get_invariants(input)
        norms = torch.cat([input[..., :1], *norms], dim=-1)
        a = unsqueeze_like(self.a, norms, dim=2)
        b = unsqueeze_like(self.b, norms, dim=2)
        norms = a * norms + b
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        return torch.sigmoid(norms) * input
