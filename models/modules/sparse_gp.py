#Copied from https://github.com/t0mnt/GTagger-experiments/blob/main/experiments/baselines/cgenn/sparse_gp.py minus certain comments

"""Custom autograd Function for the sparse geometric product (`gp_impl=sparse`).



The sparse contraction does 16x fewer MACs than the dense forms, yet measured a HIGHER
GPU peak than einsum/matmul (1.50x). Cause: what autograd RETAINS, not what the kernel
computes. The three-line eager form

    pair = x.unsqueeze(-1) * y[..., kidx]        # (B, N, 16, 16)
    w    = weight[..., spath] * spval
    out  = einsum("bnij,nij->bnj", pair, w)

saves TWO (B, N, 16, 16) tensors for backward -- `y[..., kidx]` (the mul needs it) and
`pair` (the einsum needs it) -- where einsum/matmul each save exactly one. Fewer flops,
more memory: 130.9 MB vs 81.9 MB retained per layer at the campaign shape.

lgatr 2.0 hits the same wall and solves it the same way (`_GeometricProductSparse`,
arXiv:2608.02735 §3.2): wrap the contraction in a `torch.autograd.Function` whose
`setup_context` saves only the INPUTS and whose backward recomputes the gathers. Retention
drops from two (B, N, 16, 16) tensors to two (B, N, 16) ones -- 16x smaller, and 8x below
what the dense forms retain. Their measured figure is 6.6 MB against our 130.9 MB.

Bit-identity: `Function.forward` IS `sparse_gp_expression`, so the eager Function and the
compiled direct path are bit-identical to each other by construction. Grad mode is off
inside `Function.forward`, so `pair` and the gathered `y` become transient instead of
retained -- same arithmetic, no tensors held. The BIT gate pins `gp_impl=einsum`; sparse
is TOL-class against it. Since 2026-08-16 the fc contraction inside the expression is the
race-adopted blockdiag GEMM (see `sparse_gp_expression`), which is additionally TOL-class
against the einsum spelling it replaced -- the gates in test_sparse_gp.py state the split.

Gates: tests/experiments/test_sparse_gp.py (KEEP, fixture-free) owns this module -- the
backward here is HAND-WRITTEN, so unlike the rest of the CGENN work it is not implied by
any gated forward and needs gradcheck of its own.

Determinism: the hand-written backward deliberately avoids `scatter_add_`/`index_add_`,
which are nondeterministic on CUDA and are what plain autograd uses for BOTH gathers here.
    - dL/dy: for each left blade i the map j -> k(i, j) is a BIJECTION (left multiplication
      by a fixed basis blade permutes the basis, asserted in CliffordAlgebra.__init__), so
      the scatter inverts into a gather through `algebra.gp_j_idx`.
    - dL/dweight: the (i, j) -> compact-path map is a fixed 0/+-1 matrix (35 x 256 here),
      so the segment-sum is one small GEMM instead of an index_add_.
So this is not only smaller, it makes the sparse backward deterministic on GPU, which the
autograd-composed version never was.

lgatr reached the opposite conclusion on the same question and says so in
`primitives/bilinear.py`: "index_add_ is CUDA-nondeterministic, but beats the
deterministic gather+matmul alternative by ~10% on CPU and ~5% on CUDA". Their trade is
different from ours in two ways. Their GP carries no weight, so dL/dy IS their backward
and a 10% loss there is a 10% loss overall; here it is one of three contractions. And
the gather can be placed either before or after the m-contraction -- placing it after
(what this file does) is what makes it cheap. Measured on the campaign shape, single
CPU thread, per dL/dy call:

    plain layer  index_add_ 40.8 ms   gather-late 47.1 ms   gather-early 121.2 ms
    fc layer     index_add_ 70.1 ms   gather-late 55.0 ms   gather-early 194.9 ms

"""

import torch


def sparse_gp_expression(x, y, weight, kidx, spath, spval):
    """The eager three-liner, as a function. Used verbatim in two places: inside
    `SparseGeometricProduct.forward` (where grad mode is off, so its intermediates are
    transient), and DIRECTLY on the compiled path, where the Function must not be used.

    EAGER-ONLY, and that is measured, not assumed. Under `torch.compile` AOTAutograd's
    partitioner already decides what to save across the fwd/bwd boundary, and it reaches
    the same retention this Function was written to force -- so the Function's whole
    benefit evaporates while its recompute cost stays. Model level, tag_cgenn, fp64, five
    warm-up steps then ten timed:

        posture   | expression            | Function              | ratio
        eager     | 2536 ms /  487.5 MB   | 2230 ms /  84.8 MB    | 0.88x time, 0.17x mem
        compiled  |  572 ms /  252.8 MB   | 1054 ms / 252.9 MB    | 1.84x time, 1.00x mem

    So: a clear win eager, a pure 1.84x loss compiled. `tag_cgenn.yaml` ships
    `compile: true`, which makes the compiled row the one the campaign runs -- hence the
    `torch.compiler.is_compiling()` split at the two call sites, the same compile-twin
    pattern this repo already uses for ParT's dense pair path and the static-k kNN.

    Whether the 1.84x is inherent to a custom Function under inductor or specific to this
    one is NOT established -- it was measured on CPU inductor, and CPU emits C++ where GPU
    emits Triton. The split does not depend on the answer: the expression is at least as
    fast in both postures once AOT is in charge.

    FC CONTRACTION: block-diagonal flat GEMM, not the einsum (2026-08-16). The dim-3
    spec's einsum lowers to bmm over 16 j-slices; respelling it as ONE GEMM against a
    block-diagonal weight -- 16x the MACs on paper, zero activation copies -- won the
    H100 race at every campaign shape (0.72-0.81x of einsum, fwd+bwd, CUDA events,
    `float32_matmul_precision=highest` pinned: utils/sparse_gp_race.py, verdict recorded
    in docs/cgenn-compile.md round 4). TOL-class vs the einsum spelling (reassociation
    only, ~1e-15 fp64); the einsum stays for the dim-2 layer, which was not raced. Both
    call sites (the eager Function and the compiled direct path) share this function, so
    they remain bit-identical to EACH OTHER either way.
    """
    pair = x.unsqueeze(-1) * y[..., kidx]
    w = weight[..., spath] * spval
    if weight.dim() == 3:
        m, n, nb = w.shape[0], w.shape[1], w.shape[-1]
        # w (m,n,i,j) -> (n,i,m,j) -> diagonal-embed j -> (n,i,j,m,j') -> (n*16*16, m*16)
        wbd = torch.diag_embed(w.permute(1, 2, 3, 0).movedim(2, -1))
        wbd = wbd.permute(0, 1, 3, 2, 4).reshape(n * nb * nb, m * nb)
        return (pair.reshape(pair.shape[0], -1) @ wbd).view(-1, m, nb)
    return torch.einsum(_SPECS[weight.dim()][0], pair, w)


# (forward, t-builder, dL/dweight) contraction specs, keyed by weight.dim(). The
# fully-connected layer sums over the input-feature axis `n` into an output-feature axis
# `m`; the plain layer keeps `n` as a batch axis. Same math otherwise -- the fc spec is the
# plain one with `m` added. `t[b, n, i, j] = sum_m g[b, (m), j] * w[(m), n, i, j]` is shared
# by dL/dx and dL/dy; the (i, j) -> k inversion is applied to THAT, see backward.
#
# EVERY spec here is TWO-OPERAND, and that is load-bearing, not style. torch.einsum hands
# three or more operands to opt_einsum, whose path search reads CONCRETE sizes -- under
# torch.compile(dynamic=True) that re-specializes the graph per batch shape. This repo
# already paid for that lesson once in the forward (see the einsum branch in gp.py). The
# first version of this backward used two 3-operand einsums and recompiled once per
# distinct batch shape in a compiled TRAINING step -- 1, 2, 3, 4 unique graphs over four
# shapes where the einsum impl held at 1 -- which every gate here missed, because RECOMP
# is measured under no_grad and so never sees the joint graph. Keep them binary.
_SPECS = {
    2: ("bnij,nij->bnj", "bnj,nij->bnij", "bnij,bnj->nij"),
    3: ("bnij,mnij->bmj", "bmj,mnij->bnij", "bnij,bmj->mnij"),
    # the dim-3 FORWARD spec is documentation now (the expression ships the blockdiag
    # respelling of that exact contraction); its t-builder and dL/dweight specs are live.
}
_SPEC_GX = "bnij,bnij->bni"  # same for both layer forms: m is already contracted into t
_SPEC_GY = "bni,bnik->bnk"


def _common_dtype(*tensors):
    """Widest dtype among the operands.

    Inert in the shipped posture (every model config ships `use_amp: false`, so x, y,
    weight and the incoming gradient are all the model dtype and every `.to()` below
    returns `self`). It exists because a custom Function is not autocast-aware: under
    autocast the forward einsum would emit bf16 while the saved activations stayed fp32,
    and the backward -- which runs OUTSIDE autocast -- would then feed einsum two dtypes
    and raise. Promoting once here is the cheap way to keep that door shut.
    """
    dt = tensors[0].dtype
    for t in tensors[1:]:
        dt = torch.promote_types(dt, t.dtype)
    return dt


class SparseGeometricProduct(torch.autograd.Function):
    """out[b, (m), j] = sum_(n,) i  w[(m), n, i, j] * x[b, n, i] * y[b, n, k(i, j)]

    with w[..., i, j] = weight[..., spath[i, j]] * spval[i, j] the per-grade-path weight
    scaled by the +-1 cayley entry (zero where `product_paths` masks the grade triple).

    x and y must have the SAME shape. Both call sites satisfy that by construction
    (`y = self.linear_right(x)`), so no guard is spent on it -- a shape comparison here
    would be a dynamo guard on the hot path to catch something that cannot happen. If it
    ever does, the backward returns broadcast-shaped gradients and autograd rejects them
    by shape; it fails loudly rather than silently.
    """

    generate_vmap_rule = True  # same as lgatr 2.0's _GeometricProductSparse

    @staticmethod
    def forward(x, y, weight, kidx, jinv, spath, spval, sel):
        # literally the expression it replaces -- bit-identity is the point. Grad mode is
        # off in here, so `pair` and the gathered `y` are transient rather than retained.
        return sparse_gp_expression(x, y, weight, kidx, spath, spval)

    @staticmethod
    def setup_context(ctx, inputs, output):
        # inputs only -- no intermediate is kept, which is the whole memory argument.
        # The five tables are non-persistent module buffers that outlive the step anyway,
        # so saving them costs nothing beyond the reference.
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_out):
        x, y, weight, kidx, jinv, spath, spval, sel = ctx.saved_tensors
        need_x, need_y, need_w = ctx.needs_input_grad[:3]
        out_dtypes = (x.dtype, y.dtype, weight.dtype)
        dt = _common_dtype(grad_out, x, y, weight)
        g, x, y = grad_out.to(dt), x.to(dt), y.to(dt)
        _, spec_t, spec_w = _SPECS[weight.dim()]

        w = weight.to(dt)[..., spath] * spval.to(dt)  # recomputed, not saved (one gather)
        gx = gy = gw = None
        # recomputed, not saved: this gather is the 130.9 MB half of the problem
        ygath = y[..., kidx] if (need_x or need_w) else None

        if need_w:
            # segment-sum over the (i, j) pairs sharing a compact path, as a GEMM against
            # the fixed 0/+-1 selection matrix -- deterministic on CUDA, where the
            # index_add_ autograd would emit is not.
            #
            # This one GEMM's accuracy tracks `float32_matmul_precision`, which the repo
            # ships at `highest` (config/default.yaml) -- i.e. TF32 off, full fp32.
            # Flipping it to `high` would round the gradient's mantissa to 10 bits here,
            # and unlike a real matmul there is nothing to buy for it: sel is 35 x 256 of
            # exact 0 and +-1, so this is a gather-and-add wearing a GEMM's clothes and
            # TF32 would be pure loss.
            v = x.unsqueeze(-1) * ygath  # == the forward's `pair`
            gw = torch.einsum(spec_w, v, g).flatten(-2) @ sel.to(dt).T
            del v  # free before t allocates its own (B, N, 16, 16)
        if need_x or need_y:
            t = torch.einsum(spec_t, g, w)
            if need_x:
                gx = torch.einsum(_SPEC_GX, t, ygath)
            if need_y:
                # `t` is the same contraction the index_add_ form would do; invert ITS
                # scatter with a gather through the blade bijection. Gathering late
                # matters: gathering `w` and `g` first (the obvious form) inflates the
                # contraction to (B, M, 16, 16) operands and costs 2.6x/3.5x this.
                # Bit-identical to index_add_, measured.
                gy = torch.einsum(_SPEC_GY, x, t.gather(-1, jinv.expand(t.shape)))
            del t
        del ygath

        return (
            *(None if gr is None else gr.to(d)
              for gr, d in zip((gx, gy, gw), out_dtypes)),
            None, None, None, None, None,  # kidx, jinv, spath, spval, sel
        )


def sparse_geometric_product(x, y, weight, algebra, spath, spval, sel):
    """Thin call-site wrapper: one definition of the argument order, two layers.

    Routes around the Function while compiling -- see `sparse_gp_expression` for the
    measurement. `torch.compiler.is_compiling()` is constant-folded by dynamo at trace
    time, so the compiled graph contains only the expression and this costs no break
    (GATE-BREAKS = 0). Both branches are the same arithmetic in the same order, so the
    forward is bit-identical either way and the BIT gate covers both.
    """
    if torch.compiler.is_compiling():
        return sparse_gp_expression(x, y, weight, algebra.gp_k_idx, spath, spval)
    return SparseGeometricProduct.apply(
        x, y, weight, algebra.gp_k_idx, algebra.gp_j_idx, spath, spval, sel
    )
