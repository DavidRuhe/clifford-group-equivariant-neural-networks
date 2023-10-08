import torch.nn.functional as F
from torch import nn

from algebra.cliffordalgebra import CliffordAlgebra
from engineer.metrics.metrics import Loss, MetricCollection
from models.modules.gp import SteerableGeometricProductLayer
from models.modules.linear import MVLinear


class O5CGMLP(nn.Module):
    def __init__(
        self,
        ymean, 
        ystd,
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
            MVLinear(self.algebra, 2, 8, subspaces=False),
            SteerableGeometricProductLayer(self.algebra, 8),
        )

        self.mlp = nn.Sequential(
            nn.Linear(8, 580),
            nn.ReLU(),
            nn.Linear(580, 580),
            nn.ReLU(),
            nn.Linear(580, 1),
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

        self.ymean = ymean
        self.ystd = ystd

    def _forward(self, x):
        gp = self.gp(x)
        return self.mlp(gp[..., 0])

    def forward(self, batch, step):
        points, products = batch

        points = points.view(len(points), 2, 5)
        input = self.algebra.embed_grade(points, 1)

        y = self._forward(input)
        normalized_y = y * self.ystd + self.ymean
        normalized_products = products * self.ystd + self.ymean

        assert y.shape == products.shape, breakpoint()
        loss = F.mse_loss(normalized_y, normalized_products.float(), reduction="none")
        return loss.mean(), {
            "loss": loss,
        }
