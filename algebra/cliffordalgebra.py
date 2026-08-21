# from https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks
import functools
import math

import torch
from torch import nn

from .metric import (
    ShortLexBasisBladeOrder,
    construct_gmt,
    gmt_element,
)



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
            construct_gmt(self.bbo.index_to_bitmap, self.bbo.bitmap_to_index, self.metric)
            .to_dense()
            .to(torch.get_default_dtype())
        )
        self.grades = self.bbo.grades.unique()
        # python-int copy for loops/indexing under torch.compile: int(0-d tensor) is a
        # Tensor.item() call, and item() inside the traced region is a dynamo graph break.
        self.grades_list = [int(g) for g in self.grades]
        self.register_buffer(
            "subspaces",
            torch.tensor(tuple(math.comb(self.dim, g) for g in self.grades)),
        )
        self.n_subspaces = len(self.grades)
        self.grade_to_slice = self._grade_to_slice(self.subspaces)
        # BUFFERS, not a plain list of tensors. `norms()`/`mag2s()` pass these as `blades`
        # into beta/gamma, where they index the sign BUFFERS -- and `MVSiLU.forward` and
        # `normalization.forward` both call `norms()`, so this is the live forward path.
        # A python list is invisible to `.to(device)`, so on GPU every one of those calls
        # indexed a CUDA tensor with a CPU index. Advanced indexing `gpu[cpu_idx]` is legal
        # in torch and raises nowhere -- it just copies the index host->device on every
        # forward, which is the same silent-cost class as the CliffordAlgebra.b() finding
        # (see tests/experiments/test_device_hygiene.py). Registered individually because
        # the five grades have different widths (1, 4, 6, 4, 1); non-persistent because they
        # are derived constants (aranges over the grade slices), so state_dict is unchanged.
        for _g, _s in enumerate(self.grade_to_slice):
            self.register_buffer(f"_grade_to_index_{_g}",
                                 torch.tensor(range(*_s.indices(_s.stop))), persistent=False)

        self.register_buffer("bbo_grades", self.bbo.grades.to(torch.get_default_dtype()))
        self.register_buffer("even_grades", self.bbo_grades % 2 == 0)
        self.register_buffer("odd_grades", ~self.even_grades)
        # Grade-involution sign vectors. BUFFERS, not @functools.cached_property.
        #
        # cached_property writes its value straight into instance.__dict__, bypassing
        # nn.Module.__setattr__ -- so the tensor is not a buffer, `.to(device)` cannot see
        # it, and it stays on CPU forever. `__init__` reaches all three (via _grade_to_slice
        # -> the involution helpers), so the cache is ALWAYS populated before the model is
        # moved: not an ordering hazard, an unconditional one. On GPU, `signs * mv` in
        # alpha/beta/gamma then raises "Expected all tensors to be on the same device"
        # -- and mvlayernorm -> norm -> q -> b -> beta is on the live forward path, so
        # tag_cgenn and both CGENN hybrids could not run on a GPU at all. Invisible to every
        # gate in this repo because they all run on CPU. (Pre-existing: identical in both
        # pre-dedup copies on main.)
        #
        # persistent=False keeps state_dict byte-identical: these are derived constants
        # (+/-1 per blade), fully determined by the metric, with no learnable content.
        self.register_buffer("_alpha_signs", torch.pow(-1, self.bbo_grades),
                             persistent=False)
        self.register_buffer("_beta_signs",
                             torch.pow(-1, self.bbo_grades * (self.bbo_grades - 1) // 2),
                             persistent=False)
        self.register_buffer("_gamma_signs",
                             torch.pow(-1, self.bbo_grades * (self.bbo_grades + 1) // 2),
                             persistent=False)
        self.register_buffer("cayley", cayley)
        # <x y>_0 fast path: cayley[:, 0, :] is exactly diagonal for a non-degenerate
        # metric (blade_i blade_k has a scalar component only at k == i, coefficient +-1),
        # and the diagonal restricted to any grade is that grade's metric signature. Fused
        # once with the beta involution signs, b()/q()/norm()/norms()/qs() all reduce to
        # `(x * _qb_diag * y).sum(-1)` -- no cayley gather, no einsum, no per-call index
        # tensors (see b()). Asserted rather than assumed: a degenerate metric would put
        # mass off the diagonal, and the failure mode of a silently-wrong fast path is
        # wrong physics, not a crash. Non-persistent: derived, state_dict unchanged.
        _diag0 = cayley[:, 0, :].diagonal()
        assert (cayley[:, 0, :] == torch.diag(_diag0)).all(), (
            "cayley[:, 0, :] is not diagonal: the b()/q() scalar-product fast path is "
            "invalid for this metric")
        self.register_buffer("_qb_diag", self._beta_signs * _diag0, persistent=False)
        # blade index -> subspace(grade) index, e.g. [0,1,1,1,1,2,...,4]: index_select with
        # this replaces every tensor-valued repeat_interleave over the blade dimension
        # (pure data movement -> bit-identical; tensor-valued repeats are also a dynamo
        # graph-break hazard). Non-persistent: derived data, state_dict unchanged.
        self.register_buffer(
            "blade_subspace_idx",
            torch.arange(self.n_subspaces).repeat_interleave(
                torch.tensor(tuple(math.comb(self.dim, g) for g in self.grades))),
            persistent=False,
        )
        # The blade basis is quasigroup-like: blade_i * blade_k lands on exactly ONE output
        # blade with a +-1 coefficient, so for each (left blade i, output blade j) exactly
        # one right blade k has cayley[i, j, k] != 0 (256 nonzeros of 4096). Store that k
        # and its value: the sparse gp_impl contracts only these (lgatr 2.0's sparse_gp
        # trick, adapted to CGENN's per-path weights) -- 16x fewer MACs, gather + einsum
        # only (no scatter), deterministic. Non-persistent: derived, state_dict unchanged.
        assert ((cayley != 0).sum(-1) <= 1).all(), "cayley lost the one-nonzero-per-(i,j) property"
        gp_k_idx = cayley.abs().argmax(dim=-1)
        self.register_buffer("gp_k_idx", gp_k_idx, persistent=False)
        self.register_buffer(
            "gp_val",
            torch.gather(cayley, -1, gp_k_idx.unsqueeze(-1)).squeeze(-1),
            persistent=False,
        )
        # Stronger than the assert above: for a FIXED left blade i, j -> k(i, j) is a
        # BIJECTION (left multiplication by an invertible basis blade permutes the basis).
        # That is what lets the sparse backward invert its dL/dy scatter into a gather --
        # gp_j_idx[i, k] is the j that sends (i, j) to k. Asserted rather than assumed: a
        # metric with a degenerate direction would break it, and the failure mode of a
        # silently-wrong inverse is wrong gradients, not a crash.
        _ar = torch.arange(self.n_blades).expand(self.n_blades, self.n_blades)
        assert (gp_k_idx.sort(-1).values == _ar).all(), (
            "gp_k_idx rows are not permutations: the blade basis is not quasigroup-like "
            "under this metric, so the sparse gp_impl's backward is invalid")
        # full_like(-1), not empty_like: the scatter covers every entry only BECAUSE the
        # assert above holds, and `python -O` strips asserts. Then an uncovered entry would
        # be uninitialized memory used as a gather index -- silently wrong gradients. -1
        # makes the same case an out-of-range gather, which is loud.
        self.register_buffer(
            "gp_j_idx", torch.full_like(gp_k_idx, -1).scatter_(1, gp_k_idx, _ar),
            persistent=False)
        # Grade-path table. The LAST of the four tensors that `.to(device)` could not
        # reach: it was a @functools.cached_property, which stores its value in
        # instance.__dict__ and so bypasses nn.Module.__setattr__. Its consumers only ever
        # call .nonzero()/.sum()/.size() on it at THEIR __init__ (gp.py, fcgp.py), so it
        # never crashed -- but that was a property of the call sites, not of the storage,
        # and the other three in this family all did bite. Derived entirely from cayley and
        # grade_to_slice, so persistent=False leaves state_dict byte-identical.
        #
        # Converting it also retires the warm-up loop that stood here. That loop walked the
        # MRO touching every cached_property so the compiled net would never hit the
        # descriptor's RLock (a graph break dynamo.explain cannot see, because any eager
        # warm-up fills the caches first and explain then reads plain attributes). With no
        # cached_property left in the MRO it was dead code -- and worse than dead: forcing
        # the sign vectors to materialize at __init__, BEFORE any `.to(device)`, is exactly
        # what turned upstream's latent CPU-pinning hazard into a guaranteed GPU crash here.
        # The cold-model BREAKS gate is what now catches a newly added cached_property.
        gp_paths = torch.zeros((self.dim + 1, self.dim + 1, self.dim + 1), dtype=bool)
        for i in range(self.dim + 1):
            for j in range(self.dim + 1):
                for k in range(self.dim + 1):
                    m = self.cayley[self.grade_to_slice[i],
                                    self.grade_to_slice[j],
                                    self.grade_to_slice[k]]
                    gp_paths[i, j, k] = (m != 0).any()
        self.register_buffer("geometric_product_paths", gp_paths, persistent=False)

    @property
    def grade_to_index(self):
        """Per-grade blade indices, read from the registered buffers so they follow `.to()`."""
        return [getattr(self, f"_grade_to_index_{g}") for g in range(len(self.grade_to_slice))]

    def geometric_product(self, a, b, blades=None):
        cayley = self.cayley

        if blades is not None:
            blades_l, blades_o, blades_r = blades
            assert isinstance(blades_l, torch.Tensor)
            assert isinstance(blades_o, torch.Tensor)
            assert isinstance(blades_r, torch.Tensor)
            cayley = cayley[blades_l[:, None, None], blades_o[:, None], blades_r]

        # Two-operand chain replacing the 3-operand einsum "...i,ijk,...k->...j": with
        # >= 3 operands torch.einsum computes an opt_einsum contraction path from the
        # operands' concrete sizes on every call -- python work per forward, and under
        # torch.compile(dynamic=True) it reads the symbolic batch size as an int and
        # re-specializes the graph per shape (the RECOMP gate's per-shape recompiles).
        # The chains below are exactly the paths opt_einsum selects at production sizes,
        # written out pairwise; the selector reads only blade dims, which are static.
        # Bit-identity to the old call is enforced by the cgenn_compile BIT gate.
        if cayley.shape[1] == 1 and cayley.shape[0] > 1:
            # grade-subset (g, 1, g), g > 1: GEMM-first ("ijk,bci->jkbc"; "jkbc,bck->bcj")
            t = torch.einsum("ijk,...i->jk...", cayley, a)
            return torch.einsum("jk...,...k->...j", t, b)
        # full table (16, 16, 16) and the scalar subset (1, 1, 1): outer-first
        # ("bck,bci->bcki"; "bcki,ijk->bcj")
        t = torch.einsum("...k,...i->...ki", b, a)
        return torch.einsum("...ki,ijk->...j", t, cayley)

    def _grade_to_slice(self, subspaces):
        grade_to_slice = list()
        subspaces = torch.as_tensor(subspaces)
        for grade in self.grades:
            # int endpoints: tensor-valued slice bounds call Tensor.item() (__index__) at
            # every mv[..., s] -- the last graph-break source in the compiled net
            index_start = int(subspaces[:grade].sum())
            index_end = index_start + math.comb(self.dim, grade)
            grade_to_slice.append(slice(index_start, index_end))
        return grade_to_slice

    # The three involutions were `signs * mv.clone()` upstream. The clone is dead: the mul
    # is out-of-place, so it allocates its own output and never touches `mv` -- nothing in
    # this file mutates an involution result or its argument in place (`embed`/`embed_grade`
    # index-assign into a tensor they just created). Removed: bit-identical, and these sit
    # on the live forward path via mvlayernorm -> norm -> q -> b -> beta, feeding the
    # "copy_ = 38% of runtime" finding. 0.024 ms -> 0.015 ms at (4096, 16). lgatr writes
    # its equivalent as `involution * x` with no clone.
    def alpha(self, mv, blades=None):
        signs = self._alpha_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv

    def beta(self, mv, blades=None):
        signs = self._beta_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv

    def gamma(self, mv, blades=None):
        signs = self._gamma_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv

    def zeta(self, mv):
        return mv[..., :1]

    def embed(self, tensor: torch.Tensor, tensor_index: torch.Tensor) -> torch.Tensor:
        # self.n_blades, NOT 2**self.dim -- the two are the same number by construction
        # (n_blades = len(bbo.grades) = 2**num_bases = 2**dim; asserted in __init__), but one
        # is a precomputed python int and the other is an EXPRESSION dynamo can carry into the
        # graph symbolically. When it does, inductor lowers the resulting stride as
        # `libdevice.pow(2.0, ks0)` -- a float -- and the Triton kernel fails to compile:
        #   triton_poi_fused_index_put_zeros_6 ... tl.store(out_ptr0 + (x0*(libdevice.pow(2.0, ks0))), ...)
        #   IncompatibleTypeErrorImpl('invalid operands of type pointer<fp32> and float32')
        # observed on GPU (H100, torch 2.8.0a0+nv25.08) for compiled tag_cgenn; CPU inductor
        # emits C++ and never hits it, which is why every gate in this repo was green.
        mv = torch.zeros(*tensor.shape[:-1], self.n_blades, device=tensor.device, dtype=tensor.dtype)
        mv[..., tensor_index] = tensor
        return mv

    def embed_grade(self, tensor: torch.Tensor, grade: int) -> torch.Tensor:
        mv = torch.zeros(*tensor.shape[:-1], self.n_blades, device=tensor.device)  # see embed()
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
        # The scalar-output geometric product IS a signed dot product (cayley[:, 0, :] is
        # exactly diagonal -- asserted at __init__), so <beta(x) y>_0 collapses to an
        # elementwise multiply-reduce against the fused sign vector _qb_diag. This retires
        # the einsum chain + per-call cayley gather + two per-call index-tensor
        # allocations that made b()/q() THE compiled-graph marshalling source: every
        # MVSiLU / MVLayerNorm / NormalizationLayer invariant funnels through here
        # (docs/cgenn-compile.md, Phase 1). Big eager win too (~10x on the grade-subset
        # call at production widths, CPU).
        #
        # TOL-class rewrite, NOT bit-identical to the old einsum route: identical products,
        # different reduction order (measured fp64 rel ~3e-16, fp32 ~2e-7; grades 0/1/3/4
        # came out bit-equal in the lab, g2 and the full call differ at ulp level). Folding
        # beta into the diagonal is exact: +-1 multiplies commute without rounding. The
        # BIT fixtures were re-recorded for this change.
        if blades is None:
            return (x * self._qb_diag * y).sum(-1, keepdim=True)
        assert len(blades) == 2
        if blades[0] is blades[1]:
            # every in-repo caller lands here: q() passes the SAME index object twice
            return (x * self._qb_diag[blades[0]] * y).sum(-1, keepdim=True)
        # distinct left/right blade sets (no in-repo caller): original general route
        blades = (
            blades[0].to(x.device, dtype=torch.long),
            torch.tensor([0], device=x.device, dtype=torch.long),
            blades[1].to(x.device, dtype=torch.long),
        )
        return self.geometric_product(
            self.beta(x, blades=blades[0]),
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


    @staticmethod
    def _as_int_grades(grades):
        # iterate grades as python ints OUTSIDE the traced graph: indexing python lists with
        # 0-d tensors (or int() on them) is a Tensor.item() call per element = dynamo graph
        # break. Callers inside compiled regions must pass grades_list (already ints); this
        # converts stragglers exactly once. Pure index bookkeeping -> bit-identical.
        return [g if isinstance(g, int) else int(g) for g in grades]

    def norms(self, mv, grades=None):
        if grades is None:
            grades = self.grades_list
        return [
            self.norm(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in self._as_int_grades(grades)
        ]

    def qs(self, mv, grades=None):
        if grades is None:
            grades = self.grades_list
        return [
            self.q(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in self._as_int_grades(grades)
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
        v[:, vector_indices] = torch.randn(n, vector_indices.sum(), device=self.cayley.device)
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
