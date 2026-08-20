"""Gates for the sparse geometric product (gp_impl="sparse").

WHAT EACH TEST PINS
  PARITY   sparse == the shipped einsum, fp64, for every signature the library ships.
           Two references, deliberately: the einsum this PR routes around (the
           acceptance bar) and the pair-expression the Function wraps (which isolates
           an autograd bug from an algebraic one).
  CHECK    gradcheck + gradgradcheck. gradgradcheck is not ceremony: it is what proves
           the backward is itself built from differentiable ops (no in-place fold onto
           a saved tensor, no detach), i.e. that double backward still works downstream.
  PATHS    masked grade paths carry zero and cannot leak into a gradient. The compact
           table clamps masked triples onto path index 0, where LIVE entries also sit,
           so this is the sharp edge of the construction -- and the one that varies
           most across signatures, since geometric_product_paths differs per algebra.
  SAVED    the Function retains ~1/16 of what the eager expression does. EAGER ONLY:
           under torch.compile AOTAutograd's partitioner reaches the same retention on
           its own, so this ratio does not describe the compiled path.
  REFUSED  a degenerate metric fails LOUDLY at construction rather than returning
           silently wrong gradients.
  DETERM   the CUDA backward is bitwise reproducible, where index_add_ atomics are not.

SIGNATURE COVERAGE. Everything here is built from algebra.cayley at construction;
nothing is Cl(1,3)-specific. The parametrization is what turns "works for any
signature" from a sentence into a tested claim. Valid for arbitrary (p, q) with
r = 0; degenerate algebras are refused, see test_degenerate_metric_refused.
"""

import math

import pytest
import torch

from algebra.cliffordalgebra import CliffordAlgebra, sparse_gp_tables
from models.modules.sparse_gp import sparse_geometric_product

# (metric, label). The label is the algebra Cl(p, q) for a metric of p entries
# +1 and q entries -1 -- count the entries, not the experiment name.
SIGNATURES = [
    ((1, 1), "Cl(2,0) / O(2)"),
    ((1, -1), "Cl(1,1)"),
    ((1, 1, 1), "Cl(3,0) / O(3)"),
    ((1, 1, -1), "Cl(2,1)"),
    ((1, -1, -1, -1), "Cl(1,3) -- top tagging"),
    ((1, 1, 1, 1, 1), "Cl(5,0) / O(5)"),
    ((1, 1, 1, 1, -1), "Cl(4,1)"),
    ((1, 1, 1, -1, -1), "Cl(3,2)"),
]
SIG_IDS = [lbl for _, lbl in SIGNATURES]
METRICS = [m for m, _ in SIGNATURES]

LAYERS = [False, True]
LAYER_IDS = ["gp", "fcgp"]


def _tables(alg, dtype):
    """(path_idx, spath, spval, sel) at `dtype`. spath is an INDEX tensor: converting
    it to a float dtype would corrupt it, hence the is_floating_point guard."""
    pidx = alg.geometric_product_paths.nonzero().T.contiguous()
    spath, spval, sel = (
        t.to(dtype) if t.is_floating_point() else t
        for t in sparse_gp_tables(alg, pidx)
    )
    return pidx, spath, spval, sel


def _dense_weight(alg, w_compact, pidx, out_features, in_features):
    """The dense (M, N, nb, nb, nb) weight the einsum path materialises, from the
    compact (M, N, n_paths) parameter -- i.e. FullyConnectedSteerableGeometricProduct
    ._get_weight(), reproduced here so the comparison is against the SHIPPED path."""
    w = torch.zeros(
        out_features, in_features, *alg.geometric_product_paths.size(),
        dtype=w_compact.dtype, device=w_compact.device,
    )
    w[:, :, pidx[0], pidx[1], pidx[2]] = w_compact
    bsi = alg.blade_subspace_idx
    w = w.index_select(-3, bsi).index_select(-2, bsi).index_select(-1, bsi)
    return alg.cayley * w


def _case(metric, fc, B=7, N=6, M=6, dtype=torch.float64, seed=0):
    """(under_test, einsum_ref, pair_ref, args, alg, tables) for one (signature, layer).

    Shapes stay small because the larger algebras carry 32 blades and gradcheck is
    O(n_inputs x n_outputs) jacobian evaluations.
    """
    alg = CliffordAlgebra(tuple(float(m) for m in metric)).to(dtype)
    pidx, spath, spval, sel = _tables(alg, dtype)
    nb, P = alg.n_blades, pidx.shape[1]

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, nb, dtype=dtype, generator=g)
    y = torch.randn(B, N, nb, dtype=dtype, generator=g)
    w = torch.randn(*((M, N, P) if fc else (N, P)), dtype=dtype, generator=g)
    args = (x, y, w)

    def under_test(x, y, w):
        return sparse_geometric_product(x, y, w, alg, spath, spval, sel)

    def pair_ref(x, y, w):
        """The expression the Function wraps -- same algebra, plain autograd."""
        pair = x.unsqueeze(-1) * y[..., alg.gp_k_idx]
        if fc:
            return torch.einsum("bnij,mnij->bmj", pair, w[:, :, spath] * spval)
        return torch.einsum("bnij,nij->bnj", pair, w[:, spath] * spval)

    def einsum_ref(x, y, w):
        """The SHIPPED einsum path: dense weight, two-operand chain. This is the
        acceptance bar -- 'sparse agrees with what you already ship'."""
        outer = torch.einsum("bnk,bni->bnki", y, x)
        if fc:
            dense = _dense_weight(alg, w, pidx, M, N)
            return torch.einsum("bnki,mnijk->bmj", outer, dense)
        dense = _dense_weight(alg, w.unsqueeze(0), pidx, 1, N).squeeze(0)
        return torch.einsum("bnki,nijk->bnj", outer, dense)

    return under_test, einsum_ref, pair_ref, args, alg, (pidx, spath, spval, sel)


def _rel(got, want):
    """Relative error with a guarded denominator. The `1 +` is load-bearing: on
    near-zero references a bare ratio reports absurd values (up to 24% measured on
    gradient sums) and the gate becomes noise."""
    return ((got - want).abs().max() / (1 + want.abs().max())).item()


# --------------------------------------------------------------------------- PARITY


@pytest.mark.parametrize("metric", METRICS, ids=SIG_IDS)
@pytest.mark.parametrize("fc", LAYERS, ids=LAYER_IDS)
def test_sparse_matches_einsum_fp64(metric, fc):
    """PARITY: forward agrees with the shipped einsum at fp64 roundoff, every signature.

    TOLERANCE CLASS, and it differs by layer: the non-fc contraction is the same
    operations in the same order, so it is BIT-identical to the pair expression; the
    fc contraction is reassociated (one flat GEMM against a block-diagonal weight
    instead of a bmm over j-slices), so it is TOL there. Both are TOL against the
    DENSE einsum, which sums in a third order again. Writing either as torch.equal
    against the dense path fails spuriously.
    """
    fn, einsum_ref, pair_ref, args, _, _ = _case(metric, fc)

    rel_e = _rel(fn(*args), einsum_ref(*args))
    assert rel_e < 1e-13, (
        f"forward vs the SHIPPED einsum rel={rel_e:.3e} >= 1e-13 -- beyond "
        f"reassociation scale, the sparse contraction is computing different math."
    )

    rel_p = _rel(fn(*args), pair_ref(*args))
    if fc:
        assert rel_p < 1e-13, f"fcgp: forward vs the pair expression rel={rel_p:.3e}"
    else:
        assert torch.equal(fn(*args), pair_ref(*args)), (
            "gp: forward is NOT bit-identical to the expression the Function wraps. "
            "Only what autograd retains may differ there."
        )
    print(f"GATE-PARITY[{'fcgp' if fc else 'gp'}] einsum={rel_e:.3e} pair={rel_p:.3e}")


@pytest.mark.parametrize("metric", METRICS, ids=SIG_IDS)
@pytest.mark.parametrize("fc", LAYERS, ids=LAYER_IDS)
def test_gradients_match_autograd(metric, fc):
    """PARITY: the hand-written backward agrees with plain autograd on all three grads."""
    fn, einsum_ref, _, args, _, _ = _case(metric, fc)
    g = torch.randn(einsum_ref(*args).shape, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(1))
    out = {}
    for name, f in (("ref", einsum_ref), ("Function", fn)):
        a = tuple(t.clone().requires_grad_(True) for t in args)
        f(*a).backward(g)
        out[name] = [t.grad for t in a]
    for nm, got, want in zip(("dL/dx", "dL/dy", "dL/dw"), out["Function"], out["ref"]):
        rel = _rel(got, want)
        assert rel < 1e-13, f"{nm}: rel={rel:.3e} >= 1e-13 -- backward disagrees"
        print(f"GATE-GRAD[{'fcgp' if fc else 'gp'}] {nm} rel={rel:.3e}")


# ---------------------------------------------------------------------------- CHECK


@pytest.mark.parametrize("metric", METRICS, ids=SIG_IDS)
@pytest.mark.parametrize("fc", LAYERS, ids=LAYER_IDS)
def test_gradcheck(metric, fc):
    """CHECK: analytic backward vs a numeric jacobian, plus double backward.

    gradgradcheck proves the backward has no in-place fold onto a saved tensor and no
    detach -- the property that keeps anything needing double backward working. Shapes
    are tiny: the 32-blade algebras make the jacobian expensive fast.
    """
    fn, _, _, args, _, _ = _case(metric, fc, B=2, N=2, M=2)
    args = tuple(a.requires_grad_(True) for a in args)
    assert torch.autograd.gradcheck(fn, args, raise_exception=True)
    assert torch.autograd.gradgradcheck(fn, args, raise_exception=True)
    print(f"GATE-GRADCHECK[{'fcgp' if fc else 'gp'}] analytic == numeric, dbl-bwd OK")


# ---------------------------------------------------------------------------- PATHS


@pytest.mark.parametrize("metric", METRICS, ids=SIG_IDS)
@pytest.mark.parametrize("fc", LAYERS, ids=LAYER_IDS)
def test_masked_grade_paths(metric, fc):
    """PATHS: the one branch a full algebra NEVER exercises -- sp_val == 0.

    sparse_gp_tables zeroes the weight for any (i, j) whose grade triple product_paths
    masks, and CLAMPS its path index to 0 so the gather stays in bounds. Measured across
    every signature here, the natural path set masks NOTHING (0/16, 0/64, 0/256, 0/1024),
    so that clamp is dead code in production -- exactly the kind of branch that is wrong
    when someone finally needs it. The failure mode is the clamped entries dumping their
    gradient onto path 0, silently, in dL/dweight.

    So this builds a DELIBERATELY reduced path set (every other allowed triple) and
    demands the same two things as the unmasked case: matching forward and
    autograd-matching gradients. Do NOT write this test against the natural path set --
    it passes vacuously.
    """
    dtype = torch.float64
    alg = CliffordAlgebra(tuple(float(m) for m in metric)).to(dtype)
    full = alg.geometric_product_paths.nonzero().T.contiguous()
    pidx = full[:, ::2].contiguous()  # drop every other grade path
    spath, spval, sel = (t.to(dtype) if t.is_floating_point() else t
                         for t in sparse_gp_tables(alg, pidx))

    assert (spval == 0).any(), "the reduced path set masked nothing -- test is vacuous"
    assert (spath[spval == 0] == 0).all(), "masked entries must carry clamped index 0"
    assert (spath[spval != 0] == 0).any(), (
        "no LIVE entry shares the clamped index 0, so a clobber would be invisible here"
    )
    assert torch.equal(sel[spath.reshape(-1), torch.arange(spath.numel())],
                       spval.reshape(-1)), "sel is not the transpose of (spath, spval)"

    B, N, M, nb, P = 7, 4, 4, alg.n_blades, pidx.shape[1]
    g = torch.Generator().manual_seed(0)
    args = tuple(t.requires_grad_(True) for t in (
        torch.randn(B, N, nb, dtype=dtype, generator=g),
        torch.randn(B, N, nb, dtype=dtype, generator=g),
        torch.randn(*((M, N, P) if fc else (N, P)), dtype=dtype, generator=g)))

    def eager(x, y, w):
        pair = x.unsqueeze(-1) * y[..., alg.gp_k_idx]
        if fc:
            return torch.einsum("bnij,mnij->bmj", pair, w[:, :, spath] * spval)
        return torch.einsum("bnij,nij->bnj", pair, w[:, spath] * spval)

    def under_test(x, y, w):
        return sparse_geometric_product(x, y, w, alg, spath, spval, sel)

    rel = _rel(under_test(*args), eager(*args))
    assert rel < 1e-13, f"masked forward rel={rel:.3e} -- clamped paths leak into it"

    gout = torch.randn(eager(*args).shape, dtype=dtype,
                       generator=torch.Generator().manual_seed(2))
    got = {}
    for name, f in (("eager", eager), ("Function", under_test)):
        a = tuple(t.detach().clone().requires_grad_(True) for t in args)
        f(*a).backward(gout)
        got[name] = [t.grad for t in a]
    for nm, g0, g1 in zip(("dL/dx", "dL/dy", "dL/dw"), got["Function"], got["eager"]):
        r = _rel(g0, g1)
        assert r < 1e-13, f"masked {nm}: {r:.3e} -- clamped paths leak into the gradient"
    print(f"GATE-PATHS[{'fcgp' if fc else 'gp'}] masked="
          f"{int((spval == 0).sum())}/{spval.numel()} entries, fwd rel={rel:.3e}")


# ---------------------------------------------------------------------------- SAVED


@pytest.mark.parametrize("fc", LAYERS, ids=LAYER_IDS)
def test_retains_less_than_the_expression_it_replaced(fc):
    """SAVED: the reason the Function exists, as a gate.

    The eager expression saves TWO (B, N, nb, nb) tensors for backward --
    y[..., gp_k_idx], which the mul needs, and pair, which the einsum needs. The
    Function saves only its two (B, N, nb) inputs: nb times smaller.

    EAGER ONLY. Under torch.compile AOTAutograd's partitioner reaches this retention on
    its own, so the ratio says nothing about the compiled path. Bar is 1/8 against a
    measured ~1/16 at nb=16 -- tight enough that reinstating either intermediate fails,
    loose enough not to trip on bookkeeping. Pinned to Cl(1,3) because the bar is
    nb-dependent.
    """
    fn, _, pair_ref, args, _, _ = _case((1, -1, -1, -1), fc, B=256, dtype=torch.float32)
    args = tuple(a.requires_grad_(True) for a in args)
    saved = {}
    for name, f in (("eager", pair_ref), ("Function", fn)):
        held = {}

        def pack(t, held=held):
            held[id(t)] = t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            f(*args)
        saved[name] = sum(held.values())
        print(f"GATE-SAVED[{'fcgp' if fc else 'gp'}] {name:8s} "
              f"{saved[name] / 2 ** 20:8.3f} MB retained")
    ratio = saved["Function"] / saved["eager"]
    assert ratio < 0.125, (
        f"the Function retains {ratio:.3f}x the eager expression. It exists to retain "
        f"~1/16 -- something is being saved for backward again."
    )
    print(f"GATE-SAVED[{'fcgp' if fc else 'gp'}] Function/eager = {ratio:.4f}")


# -------------------------------------------------------------------------- REFUSED


@pytest.mark.parametrize("metric", [(0, 1, 1), (1, 1, 0), (0, 1, -1)],
                         ids=["e0-first", "e0-last", "e0-mixed"])
def test_degenerate_metric_refused(metric):
    """REFUSED: a degenerate algebra fails at CONSTRUCTION, not silently at runtime.

    For a fixed left blade i, the sparse backward inverts j -> k(i, j). That map is a
    bijection iff every metric entry is nonzero (left multiplication by a basis blade is
    then a signed basis permutation). With a zero entry, the e0*e0 = 0 row has no output
    blade, argmax degenerates, and the inversion is wrong -- so the constructor must
    refuse rather than return silently wrong gradients.

    NOTE the match string must track the assert text in cliffordalgebra.__init__. If you
    reword one, reword the other; a mismatched match= turns this into a test that passes
    for the wrong reason.
    """
    with pytest.raises(AssertionError, match="permutation|degenerate|quasigroup"):
        CliffordAlgebra(tuple(float(m) for m in metric))


# --------------------------------------------------------------------------- DETERM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="determinism is a CUDA claim")
@pytest.mark.parametrize("fc", LAYERS, ids=LAYER_IDS)
def test_backward_deterministic_on_cuda(fc):
    """DETERM: the backward is bitwise reproducible on CUDA.

    The autograd backward this replaces reduces through index_add_ atomics, whose
    summation order is nondeterministic on GPU. This one reduces through a fixed gather
    and a GEMM, so it is reproducible by construction.

    Deliberately asserts only the NEW path. Asserting that the old path DIFFERS would be
    flaky -- atomics may happen to agree on any given run -- so the old path's
    nondeterminism is documented here, not tested.

    Batch is large enough that the reduction is genuinely parallel; at toy sizes
    determinism is satisfied trivially and the test proves nothing.
    """
    fn, _, _, args, _, _ = _case((1, -1, -1, -1), fc, B=512, dtype=torch.float32)
    args = tuple(t.cuda() for t in args)
    grads = []
    for _ in range(2):
        a = tuple(t.clone().requires_grad_(True) for t in args)
        out = fn(*a)
        out.backward(torch.ones_like(out))
        grads.append([t.grad for t in a])
    for nm, g0, g1 in zip(("dL/dx", "dL/dy", "dL/dw"), *grads):
        assert torch.equal(g0, g1), (
            f"{nm} is not reproducible run-to-run -- an atomic reduction is back"
        )
    print(f"GATE-DETERM[{'fcgp' if fc else 'gp'}] two backwards bit-equal")
