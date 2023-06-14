import torch.nn.functional as F
from torch import nn

from algebra.cliffordalgebra import CliffordAlgebra
from engineer.metrics.metrics import Loss, MetricCollection
from models.modules.gp import SteerableGeometricProductLayer
from models.modules.linear import MVLinear

FEATURES = 16


class ConvexHullCGMLP(nn.Module):
    def __init__(
        self,
        in_features=FEATURES,
        hidden_features=32,
        out_features=1,
        num_layers=4,
        init_normalization=0,
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
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.init_normalization = init_normalization

        self.net = nn.Sequential(
            MVLinear(self.algebra, in_features, hidden_features, subspaces=False),
            SteerableGeometricProductLayer(
                self.algebra, hidden_features, include_first_order=True
            ),
            SteerableGeometricProductLayer(
                self.algebra, hidden_features, include_first_order=True
            ),
            SteerableGeometricProductLayer(
                self.algebra, hidden_features, include_first_order=True
            ),
            SteerableGeometricProductLayer(
                self.algebra, hidden_features, include_first_order=True
            ),
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
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
        return self.net(x)

    def forward(self, batch, step):
        points, products = batch
        input = self.algebra.embed_grade(points, 1)

        y = self._forward(input)

        y = y.norm(dim=-1)
        y = self.mlp(y).squeeze(-1)

        assert y.shape == products.shape, breakpoint()
        loss = F.mse_loss(y, products, reduction="none")

        return loss.mean(), {
            "loss": loss,
        }
