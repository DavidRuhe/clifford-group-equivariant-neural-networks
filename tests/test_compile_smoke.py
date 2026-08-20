import torch
import pytest
from algebra.cliffordalgebra import CliffordAlgebra
from models.lorentz_cggnn import LorentzCGGNN   # adapt ctor args to the repo's defaults

@pytest.mark.skipif(not hasattr(torch, "compile"), reason="needs torch>=2.0")
def test_compiled_training_steps_over_varying_shapes():
    torch.manual_seed(0)
    model = LorentzCGGNN(...)                    # smallest config that constructs
    model.compile(dynamic=True)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    torch._dynamo.reset()
    from torch._dynamo.utils import counters
    losses = []
    for n_nodes in (7, 12, 9, 15, 8):            # VARYING shapes -- the whole point
        batch = make_batch(n_nodes)              # helper: random momenta + FC edges
        loss = model(*batch).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    assert counters["graph_break"] == {} or sum(counters["graph_break"].values()) == 0
    assert losses[-1] == losses[-1]              # finite; add a falls-below check if stable
