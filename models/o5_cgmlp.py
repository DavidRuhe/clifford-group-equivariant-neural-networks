import torch.nn.functional as F
from torch import nn

from algebra.cliffordalgebra import CliffordAlgebra
from engineer.metrics.metrics import Loss, MetricCollection
from models.modules.gp import SteerableGeometricProductLayer
from models.modules.linear import MVLinear


class O5CGMLP(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.algebra = CliffordAlgebra(
            (
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            )
        )

        self.gp = nn.Sequential(
            MVLinear(self.algebra, 2, 3, subspaces=False),
            SteerableGeometricProductLayer(self.algebra, 3),
        )

        self.mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        self.train_metrics = MetricCollection(
            {
                "loss": Loss(),
            },
        )

        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )

    def _forward(self, x):
        gp = self.gp(x)
        return self.mlp(gp[..., 0])

    def forward(self, batch, step):
        points, products = batch

        points = points.view(len(points), 2, 5)
        input = self.algebra.embed_grade(points, 1)

        y = self._forward(input)

        assert y.shape == products.shape, breakpoint()
        loss = F.mse_loss(y, products.float(), reduction="none")
        return loss.mean(), {
            "loss": loss,
        }
