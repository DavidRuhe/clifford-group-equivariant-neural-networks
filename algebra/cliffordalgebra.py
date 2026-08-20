import functools
import math

import torch
from torch import nn

from .metric import ShortLexBasisBladeOrder, construct_gmt, gmt_element


def sparse_gp_tables(algebra, path_idx):
    """(_sp_path, _sp_val, _sp_sel) for the sparse gp_impl: for each (left blade i, output
    blade j) the unique right blade is algebra.gp_k_idx[i, j]; the weight that entry sees is
    the compact path weight of the grade triple (g_i, g_j, g_k), or zero where product_paths
    masks the triple. One definition per self-contained file (review finding: this was
    copy-pasted at every layer).

    _sp_sel is the transpose of that map, (n_paths, n_blades**2), scaled by the same +-1
    cayley value: `dL/dweight = einsum(...).flatten(-2) @ _sp_sel.T` is the segment-sum
    that gathers each compact path's (i, j) entries back together. Autograd would spell
    that as an index_add_, which is nondeterministic on CUDA; the GEMM is not, and at 35 x
    256 it is free. Used by sparse_gp.SparseGeometricProduct.backward.
    """
    g = algebra.bbo_grades.long()
    lookup = torch.full((algebra.n_subspaces,) * 3, -1, dtype=torch.long)
    lookup[path_idx[0], path_idx[1], path_idx[2]] = torch.arange(path_idx.shape[1])
    p = lookup[g[:, None], g[None, :], g[algebra.gp_k_idx]]
    sp_path, sp_val = p.clamp(min=0), algebra.gp_val * (p >= 0)
    sel = torch.zeros(path_idx.shape[1], sp_path.numel(), dtype=sp_val.dtype)
    # columns are unique by construction, so masked triples write their own zero and
    # cannot clobber a live entry that happens to share the clamped path index 0
    sel[sp_path.reshape(-1), torch.arange(sp_path.numel())] = sp_val.reshape(-1)
    return sp_path, sp_val, sel


class CliffordAlgebra(nn.Module):
    def __init__(self, metric):
        super().__init__()

        self.register_buffer("metric", torch.as_tensor(metric))
        self.num_bases = len(metric)
        self.bbo = ShortLexBasisBladeOrder(self.num_bases)
        self.dim = len(self.metric)
        self.n_blades = len(self.bbo.grades)
        cayley = (
            construct_gmt(
                self.bbo.index_to_bitmap, self.bbo.bitmap_to_index, self.metric
            )
            .to_dense()
            .to(torch.get_default_dtype())
        )
        self.grades = self.bbo.grades.unique()
        self.register_buffer(
            "subspaces",
            torch.tensor(tuple(math.comb(self.dim, g) for g in self.grades)),
        )
        self.n_subspaces = len(self.grades)
        self.grade_to_slice = self._grade_to_slice(self.subspaces)
        for _g, _s in enumerate(self.grade_to_slice):
            self.register_buffer(
                f"_grade_to_index_{_g}",
                torch.tensor(range(*_s.indices(_s.stop))),
                persistent=False,
            )

        self.register_buffer(
            "bbo_grades", self.bbo.grades.to(torch.get_default_dtype())
        )
        self.register_buffer("even_grades", self.bbo_grades % 2 == 0)
        self.register_buffer("odd_grades", ~self.even_grades)
        self.register_buffer("cayley", cayley)
        self.register_buffer(
            "_alpha_signs", torch.pow(-1, self.bbo_grades), persistent=False
        )
        self.register_buffer(
            "_beta_signs",
            torch.pow(-1, self.bbo_grades * (self.bbo_grades - 1) // 2),
            persistent=False,
        )
        self.register_buffer(
            "_gamma_signs",
            torch.pow(-1, self.bbo_grades * (self.bbo_grades + 1) // 2),
            persistent=False,
        )
                # ---- sparse-GP support: one-output-blade structure + the blade bijection ----
        # invariant 1: at most one nonzero output blade per (i, j) -- any diagonal metric
        assert ((cayley != 0).sum(-1) <= 1).all(), \
            "cayley lost the one-nonzero-per-(i,j) property"
        gp_k_idx = cayley.abs().argmax(dim=-1)                    # k(i, j)
        self.register_buffer("gp_k_idx", gp_k_idx, persistent=False)
        self.register_buffer(
            "gp_sign",
            torch.gather(cayley, -1, gp_k_idx.unsqueeze(-1)).squeeze(-1),
            persistent=False,
        )
        # invariant 2: for fixed i, j -> k(i, j) is a bijection. HOLDS iff the metric has
        # no zero entries (left mult by a basis blade is then a signed basis permutation).
        # Degenerate algebras (PGA etc.) FAIL HERE, on purpose: the sparse backward's
        # dL/dy inverts this map, so those algebras must keep gp_impl="einsum".
        _ar = torch.arange(self.n_blades)
        assert (gp_k_idx.sort(-1).values == _ar).all(), (
            "metric has a zero entry (degenerate algebra): j->k is not a bijection, "
            "gp_impl='sparse' is unsupported here -- use the default einsum")
        self.register_buffer("gp_j_idx", gp_k_idx.argsort(-1), persistent=False)

    def geometric_product(self, a, b, blades=None):
        cayley = self.cayley

        if blades is not None:
            blades_l, blades_o, blades_r = blades
            assert isinstance(blades_l, torch.Tensor)
            assert isinstance(blades_o, torch.Tensor)
            assert isinstance(blades_r, torch.Tensor)
            cayley = cayley[blades_l[:, None, None], blades_o[:, None], blades_r]

        return torch.einsum("...i,ijk,...k->...j", a, cayley, b)

    def _grade_to_slice(self, subspaces):
        grade_to_slice = list()
        subspaces = torch.as_tensor(subspaces)
        for grade in self.grades:
            index_start = subspaces[:grade].sum()
            index_end = index_start + math.comb(self.dim, grade)
            grade_to_slice.append(slice(index_start, index_end))
        return grade_to_slice

    @property
    def grade_to_index(self):
        """Per-grade blade indices, read from the buffers so they follow `.to(device)`."""
        return [
            getattr(self, f"_grade_to_index_{g}")
            for g in range(len(self.grade_to_slice))
        ]

    def alpha(self, mv, blades=None):
        signs = self._alpha_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def beta(self, mv, blades=None):
        signs = self._beta_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def gamma(self, mv, blades=None):
        signs = self._gamma_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def zeta(self, mv):
        return mv[..., :1]

    def embed(self, tensor: torch.Tensor, tensor_index: torch.Tensor) -> torch.Tensor:
        mv = torch.zeros(
            *tensor.shape[:-1], 2**self.dim, device=tensor.device, dtype=tensor.dtype
        )
        mv[..., tensor_index] = tensor
        return mv

    def embed_grade(self, tensor: torch.Tensor, grade: int) -> torch.Tensor:
        mv = torch.zeros(*tensor.shape[:-1], 2**self.dim, device=tensor.device)
        s = self.grade_to_slice[grade]
        mv[..., s] = tensor
        return mv

    def get(self, mv: torch.Tensor, blade_index: tuple[int]) -> torch.Tensor:
        blade_index = tuple(blade_index)
        return mv[..., blade_index]

    def get_grade(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        s = self.grade_to_slice[grade]
        return mv[..., s]

    def b(self, x, y, blades=None):
        if blades is not None:
            assert len(blades) == 2
            beta_blades = blades[0]
              blades = (
                blades[0],
                torch.tensor([0], device=self.cayley.device),   # was: torch.tensor([0])
                blades[1],
            )
        else:
            blades = torch.arange(self.n_blades, device=self.cayley.device)  # was: torch.tensor(range(...))
            blades = (
                blades,
                torch.tensor([0], device=self.cayley.device),   # was: torch.tensor([0])
                blades,
            )
            beta_blades = None

        return self.geometric_product(
            self.beta(x, blades=beta_blades),
            y,
            blades=blades,
        )

    def q(self, mv, blades=None):
        if blades is not None:
            blades = (blades, blades)
        return self.b(mv, mv, blades=blades)

    def _smooth_abs_sqrt(self, input, eps=1e-16):
        return (input**2 + eps) ** 0.25

    def norm(self, mv, blades=None):
        return self._smooth_abs_sqrt(self.q(mv, blades=blades))

    def norms(self, mv, grades=None):
        if grades is None:
            grades = self.grades
        return [
            self.norm(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

    def qs(self, mv, grades=None):
        if grades is None:
            grades = self.grades
        return [
            self.q(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

    def sandwich(self, u, v, w):
        return self.geometric_product(self.geometric_product(u, v), w)

    def output_blades(self, blades_left, blades_right):
        blades = []
        for blade_left in blades_left:
            for blade_right in blades_right:
                bitmap_left = self.bbo.index_to_bitmap[blade_left]
                bitmap_right = self.bbo.index_to_bitmap[blade_right]
                bitmap_out, _ = gmt_element(bitmap_left, bitmap_right, self.metric)
                index_out = self.bbo.bitmap_to_index[bitmap_out]
                blades.append(index_out)

        return torch.tensor(blades)

    def random(self, n=None):
        if n is None:
            n = 1
        return torch.randn(n, self.n_blades)

    def random_vector(self, n=None):
        if n is None:
            n = 1
        vector_indices = self.bbo_grades == 1
        v = torch.zeros(n, self.n_blades, device=self.cayley.device)
        v[:, vector_indices] = torch.randn(
            n, vector_indices.sum(), device=self.cayley.device
        )
        return v

    def parity(self, mv):
        is_odd = torch.all(mv[..., self.even_grades] == 0)
        is_even = torch.all(mv[..., self.odd_grades] == 0)

        if is_odd ^ is_even:  # exclusive or (xor)
            return is_odd
        else:
            raise ValueError("This is not a homogeneous element.")

    def eta(self, w):
        return (-1) ** self.parity(w)

    def alpha_w(self, w, mv):
        return self.even_grades * mv + self.eta(w) * self.odd_grades * mv

    def inverse(self, mv, blades=None):
        mv_ = self.beta(mv, blades=blades)
        return mv_ / self.q(mv)

    def rho(self, w, mv):
        """Applies the versor w action to mv."""
        return self.sandwich(w, self.alpha_w(w, mv), self.inverse(w))

    def reduce_geometric_product(self, inputs):
        return functools.reduce(self.geometric_product, inputs)

    def versor(self, order=None, normalized=True):
        if order is None:
            order = self.dim if self.dim % 2 == 0 else self.dim - 1
        vectors = self.random_vector(order)
        versor = self.reduce_geometric_product(vectors[:, None])
        if normalized:
            versor = versor / self.norm(versor)[..., :1]
        return versor

    def rotor(self):
        return self.versor()

    @functools.cached_property
    def geometric_product_paths(self):
        gp_paths = torch.zeros((self.dim + 1, self.dim + 1, self.dim + 1), dtype=bool)

        for i in range(self.dim + 1):
            for j in range(self.dim + 1):
                for k in range(self.dim + 1):
                    s_i = self.grade_to_slice[i]
                    s_j = self.grade_to_slice[j]
                    s_k = self.grade_to_slice[k]

                    m = self.cayley[s_i, s_j, s_k]
                    gp_paths[i, j, k] = (m != 0).any()

        return gp_paths
