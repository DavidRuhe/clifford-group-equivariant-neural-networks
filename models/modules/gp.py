import math

import torch
from torch import nn

from .linear import MVLinear
from .normalization import NormalizationLayer


class SteerableGeometricProductLayer(nn.Module):
    def __init__(
        self, algebra, features, include_first_order=True, normalization_init=0
    ):
        super().__init__()

        self.algebra = algebra
        self.features = features
        self.include_first_order = include_first_order

        if normalization_init is not None:
            self.normalization = NormalizationLayer(
                algebra, features, normalization_init
            )
        else:
            self.normalization = nn.Identity()
        self.linear_right = MVLinear(algebra, features, features, bias=False)
        if include_first_order:
            self.linear_left = MVLinear(algebra, features, features, bias=True)

        self.product_paths = algebra.geometric_product_paths
        self.register_buffer("_path_idx", self.product_paths.nonzero().T.contiguous(),
                             persistent=False)
        self.weight = nn.Parameter(torch.empty(features, self.product_paths.sum()))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / (math.sqrt(self.algebra.dim + 1)))

    def _get_weight(self):
        weight = torch.zeros(
            self.features,
            *self.product_paths.size(),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        weight[:, self._path_idx[0], self._path_idx[1], self._path_idx[2]] = self.weight
        bsi = self.algebra.blade_subspace_idx
        weight_repeated = (
            weight.index_select(-3, bsi).index_select(-2, bsi).index_select(-1, bsi)
        )
        return self.algebra.cayley * weight_repeated

    def forward(self, input):
        input_right = self.linear_right(input)
        input_right = self.normalization(input_right)

        weight = self._get_weight()

        if self.include_first_order:
            outer = torch.einsum("bnk,bni->bnki", input_right, input)
            return (
                self.linear_left(input)
                + torch.einsum("bnki,nijk->bnj", outer, weight)
            ) / math.sqrt(2)

        else:
            outer = torch.einsum("bnk,bni->bnki", input_right, input)
            return torch.einsum("bnki,nijk->bnj", outer, weight)
