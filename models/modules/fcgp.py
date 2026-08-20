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
        gp_imp
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
            
        self.gp_impl = gp_impl
        self.product_paths = algebra.geometric_product_paths
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, self.product_paths.sum())
        )
        if gp_impl == "sparse":
            from algebra.cliffordalgebra import sparse_gp_tables
            path_idx = self.product_paths.nonzero().T.contiguous()
            sp_path, sp_val, sp_sel = sparse_gp_tables(algebra, path_idx)
            self.register_buffer("_sp_path", sp_path, persistent=False)
            self.register_buffer("_sp_val", sp_val, persistent=False)
            self.register_buffer("_sp_sel", sp_sel, persistent=False)
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

    def forward(self, input, input_right=None, left=None):
        if input_right is None:
            input_right = self.linear_right(input)
        input_right = self.normalization(input_right)
    
        if self.gp_impl == "sparse":
            from models.modules.sparse_gp import sparse_geometric_product
            product = sparse_geometric_product(
                input, input_right, self.weight,
                self.algebra, self._sp_path, self._sp_val, self._sp_sel)
        else:
            weight = self._get_weight()
            outer = torch.einsum("bnk,bni->bnki", input_right, input)
            product = torch.einsum("bnki,mnijk->bmj", outer, weight)
    
        if self.include_first_order:
            if left is None:
                left = self.linear_left(input)
            return (left + product) / math.sqrt(2)
        return product
