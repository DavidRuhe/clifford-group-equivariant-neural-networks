import math

import torch
from torch import nn

from .utils import unsqueeze_like


class MVLinear(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        out_features,
        subspaces=True,
        bias=True,
    ):
        super().__init__()

        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.subspaces = subspaces

        if subspaces:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, algebra.n_subspaces)
            )
            self._forward = self._forward_subspaces
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(1, out_features, 1))
            self.b_dims = (0,)
        else:
            self.register_parameter("bias", None)
            self.b_dims = ()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _forward(self, input):
        return torch.einsum("bm...i, nm->bn...i", input, self.weight)

    def _forward_subspaces(self, input):
        weight = self.weight.repeat_interleave(self.algebra.subspaces, dim=-1)
        return torch.einsum("bm...i, nmi->bn...i", input, weight)

    def forward(self, input):
        result = self._forward(input)

        if self.bias is not None:
            bias = self.algebra.embed(self.bias, self.b_dims)
            result += unsqueeze_like(bias, result, dim=2)
        return result


# EPS = 1e-6

# def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor, dim=0):
#     """
#     Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

#     Args:
#         tensor (torch.Tensor): tensor to unsqueeze
#         like (torch.Tensor): tensor whose dimensions to match
#         dim: int: starting dim, default: 0.
#     """
#     n_unsqueezes = like.ndim - tensor.ndim
#     if n_unsqueezes < 0:
#         raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
#     elif n_unsqueezes == 0:
#         return tensor
#     else:
#         return tensor[dim * (slice(None),) + (None,) * n_unsqueezes]


# def get_geometric_product_paths(algebra):
#     gp_paths = torch.zeros(
#         (algebra.dim + 1, algebra.dim + 1, algebra.dim + 1), dtype=bool
#     )

#     for i in range(algebra.dim + 1):
#         for j in range(algebra.dim + 1):
#             for k in range(algebra.dim + 1):
#                 s_i = algebra.grade_to_slice[i]
#                 s_j = algebra.grade_to_slice[j]
#                 s_k = algebra.grade_to_slice[k]

#                 m = algebra.cayley[s_i, s_j, s_k]
#                 gp_paths[i, j, k] = (m != 0).any()

#     return gp_paths


# class NormalizationLayer(nn.Module):
#     def __init__(self, algebra, features, init: float = 0):
#         super().__init__()
#         self.algebra = algebra
#         self.in_features = features

#         self.a = nn.Parameter(torch.zeros(self.in_features, algebra.n_subspaces) + init)

#     def _norms(self, input):
#         slice = self.algebra.grade_to_slice
#         index = self.algebra.grade_to_index

#         return [
#             self.algebra.norm(input[..., slice[g]], blades=index[g])
#             for g in self.algebra.grades
#         ]

#     def forward(self, input):
#         assert input.shape[1] == self.in_features

#         # norms = torch.cat(
#         #     [
#         #         s.norm(dim=-1, keepdim=True)
#         #         for s in input.split(tuple(self.algebra.subspaces), dim=-1)
#         #     ],
#         #     dim=-1,
#         # )
#         # norms = self.algebra.norm(self.algebra.split_subspaces(input))[..., 0]
#         # norms = self._norms(input)
#         # breakpoint()
#         norms = torch.cat(self.algebra.norms(input), dim=-1)
#         s_a = torch.sigmoid(self.a)
#         norms = s_a * (norms - 1) + 1  # Interpolates between 1 and the norm.
#         norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
#         normalized = input / (norms + EPS)

#         return normalized


# class SteerableGeometricProductLayer(nn.Module):
#     def __init__(
#         self, algebra, features, include_first_order=True, normalization_init=0
#     ):
#         super().__init__()

#         self.algebra = algebra
#         self.features = features
#         self.include_first_order = include_first_order

#         if normalization_init is not None:
#             self.normalization = NormalizationLayer(
#                 algebra, features, normalization_init
#             )
#         else:
#             self.normalization = nn.Identity()
#         self.linear_right = MVLinear(algebra, features, features, bias=False)
#         if include_first_order:
#             self.linear_left = MVLinear(algebra, features, features, bias=True)

#         self.product_paths = get_geometric_product_paths(algebra)
#         self.weight = nn.Parameter(torch.empty(features, self.product_paths.sum()))

#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.normal_(self.weight, std=1 / (math.sqrt(self.algebra.dim + 1)))

#     def _get_weight(self):
#         weight = torch.zeros(
#             self.features,
#             *self.product_paths.size(),
#             dtype=self.weight.dtype,
#             device=self.weight.device,
#         )
#         weight[:, self.product_paths] = self.weight
#         subspaces = self.algebra.subspaces
#         weight_repeated = (
#             weight.repeat_interleave(subspaces, dim=-3)
#             .repeat_interleave(subspaces, dim=-2)
#             .repeat_interleave(subspaces, dim=-1)
#         )
#         return self.algebra.cayley * weight_repeated

#     def _contract(self, input, input_right, weight):
#         input_expanded = input.unsqueeze(3).unsqueeze(4)  # Shape: (b, n, i, 1, 1)
#         weight_expanded = weight.unsqueeze(0)  # Shape: (1, n, i, j, k)
#         temp = input_expanded * weight_expanded  # Shape: (b, n, i, j, k)
#         temp = torch.sum(temp, dim=2)  # Shape: (b, n, j, k)
#         result = torch.matmul(temp, input_right.unsqueeze(3)).squeeze(
#             3
#         )  # Shape: (b, n, j)
#         return result

#     def forward(self, input):
#         input_right = self.linear_right(input)
#         input_right = self.normalization(input_right)

#         weight = self._get_weight()

#         if self.include_first_order:
#             return (
#                 self.linear_left(input)
#                 # + torch.einsum("bni, nijk, bnk -> bnj", input, weight, input_right)
#                 + contract("bni, nijk, bnk -> bnj", input, weight, input_right)
#                 # + self._contract(input, input_right, weight)
#             ) / math.sqrt(2)

#         else:
#             # return self._contract(input, input_right, weight)
#             return contract("bni, nijk, bnk -> bnj", input, weight, input_right)


# class FullyConnectedSteerableGeometricProductLayer(nn.Module):
#     def __init__(
#         self,
#         algebra,
#         in_features,
#         out_features,
#         include_first_order=True,
#         normalization_init=0,
#     ):
#         super().__init__()

#         self.algebra = algebra
#         self.in_features = in_features
#         self.out_features = out_features
#         self.include_first_order = include_first_order

#         if normalization_init is not None:
#             self.normalization = NormalizationLayer(
#                 algebra, in_features, normalization_init
#             )
#         else:
#             self.normalization = nn.Identity()
#         self.linear_right = MVLinear(algebra, in_features, in_features, bias=False)
#         if include_first_order:
#             self.linear_left = MVLinear(algebra, in_features, out_features, bias=True)

#         self.product_paths = get_geometric_product_paths(algebra)
#         self.weight = nn.Parameter(
#             torch.empty(out_features, in_features, self.product_paths.sum())
#         )

#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.normal_(
#             self.weight,
#             std=1 / math.sqrt(self.in_features * (self.algebra.dim + 1)),
#         )

#     def _get_weight(self):
#         weight = torch.zeros(
#             self.out_features,
#             self.in_features,
#             *self.product_paths.size(),
#             dtype=self.weight.dtype,
#             device=self.weight.device,
#         )
#         weight[:, :, self.product_paths] = self.weight
#         subspaces = self.algebra.subspaces
#         weight_repeated = (
#             weight.repeat_interleave(subspaces, dim=-3)
#             .repeat_interleave(subspaces, dim=-2)
#             .repeat_interleave(subspaces, dim=-1)
#         )
#         return self.algebra.cayley * weight_repeated

#     def _contract(self, input, input_right, weight):
#         input_expanded = (
#             input.unsqueeze(1).unsqueeze(4).unsqueeze(5)
#         )  # Shape: (b, 1, n, i, 1, 1)
#         weight_expanded = weight.unsqueeze(0)  # Shape: (1, m, n, i, j, k)
#         temp = input_expanded * weight_expanded  # Shape: (b, m, n, i, j, k)
#         temp = torch.sum(temp, dim=3)  # Shape: (b, m, n, j, k)
#         temp = torch.matmul(temp, input_right.unsqueeze(1).unsqueeze(4)).squeeze(
#             4
#         )  # Shape: (b, m, n, j)
#         result_rewrite = torch.sum(temp, dim=2)  # Shape: (b, m, j)
#         return result_rewrite

#     def forward(self, input):
#         input_right = self.linear_right(input)
#         input_right = self.normalization(input_right)

#         weight = self._get_weight()

#         if self.include_first_order:
#             return (
#                 self.linear_left(input)
#                 # + torch.einsum("bni, mnijk, bnk -> bmj", input, weight, input_right)
#                 + contract("bni, mnijk, bnk -> bmj", input, weight, input_right)
#                 # + self._contract(input, input_right, weight)
#             ) / math.sqrt(2)
#         else:
#             # return self._contract(input, input_right, weight)
#             return contract("bni, mnijk, bnk -> bmj", input, weight, input_right)


# class MVSiLU(nn.Module):
#     def __init__(self, algebra, channels, invariant="norm", exclude_dual=False):
#         super().__init__()
#         self.algebra = algebra
#         self.channels = channels
#         self.exclude_dual = exclude_dual
#         self.invariant = invariant
#         self.a = nn.Parameter(torch.ones(1, channels, algebra.dim + 1))
#         self.b = nn.Parameter(torch.zeros(1, channels, algebra.dim + 1))

#         if invariant == "norm":
#             self._get_invariants = self._norms_except_scalar
#         elif invariant == "mag2":
#             self._get_invariants = self._mag2s_except_scalar
#         else:
#             raise ValueError(f"Invariant {invariant} not recognized.")

#     def _norms(self, input):
#         raise NotImplementedError
#         return torch.cat(
#             [input[..., :1]]
#             + [
#                 s.norm(dim=-1, keepdim=True)
#                 for s in input[..., 1:].split(tuple(self.algebra.subspaces)[1:], dim=-1)
#             ],
#             dim=-1,
#         )

#     def _norms_exclude_dual(self, input):
#         raise NotImplementedError
#         return torch.cat(
#             [input[..., :1]]
#             + [
#                 s.norm(dim=-1, keepdim=True)
#                 for s in input[..., 1:-1].split(
#                     tuple(self.algebra.subspaces)[1:-1], dim=-1
#                 )
#             ]
#             + [input[..., -1:]],
#             dim=-1,
#         )

#     def _norms_except_scalar(self, input):
#         return self.algebra.norms(input, grades=self.algebra.grades[1:])

#     def _mag2s_except_scalar(self, input):
#         return self.algebra.mag2s(input, grades=self.algebra.grades[1:])

#     def forward(self, input):
#         # norms = self.algebra.norm(self.algebra.split_subspaces(input)[..., 1:, :])[
#         #     ..., 0
#         # ]
#         # norms = torch.cat([input[..., :1], norms], dim=-1)
#         # a = unsqueeze_like(self.a, norms, dim=2)
#         # b = unsqueeze_like(self.b, norms, dim=2)
#         # norms = a * norms + b
#         # norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
#         # return torch.sigmoid(norms) * input * math.sqrt(3 / 2)
#         #         norms = split_subspaces(x, self.algebra)

#         # return input
#         # norms = self.algebra.split_subspaces(input)
#         # norms = norms[..., 1:, :]
#         # norms = self.algebra.norm(norms)[..., 0]

#         norms = self._get_invariants(input)
#         norms = torch.cat([input[..., :1], *norms], dim=-1)
#         a = unsqueeze_like(self.a, norms, dim=2)
#         b = unsqueeze_like(self.b, norms, dim=2)
#         norms = a * norms + b
#         norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
#         return torch.sigmoid(norms) * input


# # class MVLinear(nn.Module):
# #     def __init__(self, algebra, in_features, out_features, bias=True, dual_bias=False):
# #         super().__init__()

# #         self.algebra = algebra
# #         self.in_features = in_features
# #         self.out_features = out_features
# #         self.dual_bias = dual_bias

# #         self.weight = nn.Parameter(torch.empty(out_features, in_features, algebra.n_subspaces))

# #         if bias:
# #             self.bias = nn.Parameter(torch.empty(1, out_features, 1 + dual_bias))
# #             self.b_dims = (0, -1) if self.dual_bias else (0,)
# #         else:
# #             self.register_parameter("bias", None)
# #             self.b_dims = ()

# #         self.reset_parameters()


# #     def reset_parameters(self):
# #         torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))

# #         if self.bias is not None:
# #             # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
# #             # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
# #             # torch.nn.init.uniform_(self.bias, -bound, bound)

# #             torch.nn.init.zeros_(self.bias)


# #     def _get_weight(self):
# #         return self.weight.repeat_interleave(self.algebra.subspaces, dim=-1)


# #     def forward(self, input):
# #         weight = self._get_weight()
# #         result = torch.einsum("bm...i, nmi->bn...i", input, weight)
# #         if self.bias is not None:
# #             bias = self.algebra.embed(self.bias, self.b_dims)
# #             result += unsqueeze_like(bias, result, dim=2)
# #         return result


# class MVLayerNorm(nn.Module):
#     def __init__(self, algebra, channels):
#         super().__init__()
#         self.algebra = algebra
#         self.channels = channels
#         self.a = nn.Parameter(torch.ones(1, channels))

#     def forward(self, input):
#         norm = self.algebra.norm(input)[..., :1].mean(dim=1, keepdim=True) + EPS
#         a = unsqueeze_like(self.a, norm, dim=2)
#         return a * input / norm
#         # norms = a / (self._norms(input).mean(dim=1, keepdim=True) + EPS)
#         # return input * norms.repeat_interleave(self.algebra.subspaces, dim=-1)
