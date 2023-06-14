import math

import torch
from torch import nn

from .linear import MVLinear
from .normalization import NormalizationLayer


class FullyConnectedSteerableGeometricProductLayer(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        out_features,
        include_first_order=True,
        normalization_init=0,
    ):
        super().__init__()

        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.include_first_order = include_first_order

        if normalization_init is not None:
            self.normalization = NormalizationLayer(
                algebra, in_features, normalization_init
            )
        else:
            self.normalization = nn.Identity()
        self.linear_right = MVLinear(algebra, in_features, in_features, bias=False)
        if include_first_order:
            self.linear_left = MVLinear(algebra, in_features, out_features, bias=True)

        self.product_paths = algebra.geometric_product_paths
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, self.product_paths.sum())
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(
            self.weight,
            std=1 / math.sqrt(self.in_features * (self.algebra.dim + 1)),
        )

    def _get_weight(self):
        weight = torch.zeros(
            self.out_features,
            self.in_features,
            *self.product_paths.size(),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        weight[:, :, self.product_paths] = self.weight
        subspaces = self.algebra.subspaces
        weight_repeated = (
            weight.repeat_interleave(subspaces, dim=-3)
            .repeat_interleave(subspaces, dim=-2)
            .repeat_interleave(subspaces, dim=-1)
        )
        return self.algebra.cayley * weight_repeated

    def forward(self, input):
        input_right = self.linear_right(input)
        input_right = self.normalization(input_right)

        weight = self._get_weight()

        if self.include_first_order:
            return (
                self.linear_left(input)
                + torch.einsum("bni, mnijk, bnk -> bmj", input, weight, input_right)
            ) / math.sqrt(2)
        else:
            return torch.einsum("bni, mnijk, bnk -> bmj", input, weight, input_right)
