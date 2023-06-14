import torch.nn.functional as F
from torch import nn

from algebra.cliffordalgebra import CliffordAlgebra
from engineer.metrics.metrics import Loss, MetricCollection
from models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer
from models.modules.mvsilu import MVSiLU


class O3CGMLP(nn.Module):
    def __init__(
        self,
        in_features=3,
        hidden_features=32,
        out_features=1,
        num_layers=6,
        init_normalization=0,
    ):
        super().__init__()

        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0))
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.init_normalization = init_normalization

        net = [
            FullyConnectedSteerableGeometricProductLayer(
                self.algebra, in_features, hidden_features
            ),
        ]
        for _ in range(num_layers - 1):
            net.append(MVSiLU(self.algebra, hidden_features))

        net.append(
            FullyConnectedSteerableGeometricProductLayer(
                self.algebra, hidden_features, out_features
            )
        )

        self.net = nn.Sequential(*net)

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
        y = y[:, 0, -1]
        loss = F.mse_loss(y, products, reduction="none")

        return (
            loss.mean(),
            {
                "loss": loss,
            },
        )
