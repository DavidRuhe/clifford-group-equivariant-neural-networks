import torch
from torch import nn

from algebra.cliffordalgebra import CliffordAlgebra
from engineer.metrics.metrics import (Accuracy, LorentzMetric, Loss,
                                      MetricCollection)
from models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer
from models.modules.gp import SteerableGeometricProductLayer
from models.modules.linear import MVLinear
from models.modules.mvlayernorm import MVLayerNorm
from models.modules.mvsilu import MVSiLU


def get_invariants(algebra, input):
    norms = algebra.qs(input, grades=algebra.grades[1:])
    return torch.cat([input[..., :1], *norms], dim=-1)


def psi(p):
    """`\psi(p) = Sgn(p) \cdot \log(|p| + 1)`"""
    return torch.sign(p) * torch.log(torch.abs(p) + 1)


def unsorted_segment_sum(data, segment_ids, num_segments):
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Adapted from https://github.com/vgsatorras/egnn.
    """
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    Adapted from https://github.com/vgsatorras/egnn.
    """
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class CGLayer(nn.Module):
    def __init__(
        self,
        algebra,
        in_features_x,
        hidden_features_x,
        out_features_x,
        in_features_h,
        hidden_features_h,
        out_features_h,
        edge_attr_x=3,
        edge_attr_h=0,
        node_attr_x=2,
        node_attr_h=2,
        aggregation="mean",
        use_invariants_to_update=True,
        residual=False,
        normalization_init=None,
        layer_type="fc",
    ):
        super().__init__()
        self.edge_attr_x = edge_attr_x
        self.algebra = algebra
        invariants_h = out_features_x * self.algebra.n_subspaces

        f_in_h = 3 * in_features_h
        self.phi_h = nn.Sequential(
            nn.Linear(
                f_in_h + edge_attr_h + invariants_h,
                hidden_features_h,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
        )

        f_in_x = 3 * in_features_x
        if layer_type == "fc":
            self.phi_x = nn.Sequential(
                FullyConnectedSteerableGeometricProductLayer(
                    self.algebra,
                    edge_attr_x + f_in_x,
                    hidden_features_x,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, hidden_features_x),
            )

            self.theta_x = nn.Sequential(
                FullyConnectedSteerableGeometricProductLayer(
                    self.algebra,
                    node_attr_x + in_features_x + hidden_features_x,
                    out_features_x,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, out_features_x),
            )
        elif layer_type == "gpmlp":
            self.phi_x = nn.Sequential(
                MVLinear(self.algebra, edge_attr_x + f_in_x, hidden_features_x),
                MVSiLU(self.algebra, hidden_features_x),
                SteerableGeometricProductLayer(
                    self.algebra,
                    hidden_features_x,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, hidden_features_x),
            )
            self.theta_x = nn.Sequential(
                MVLinear(
                    self.algebra,
                    node_attr_x + in_features_x + hidden_features_x,
                    out_features_x,
                ),
                MVSiLU(self.algebra, out_features_x),
                SteerableGeometricProductLayer(
                    self.algebra,
                    out_features_x,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, out_features_x),
            )
        else:
            raise ValueError(f"Unknown layer type {layer_type}.")

        self.theta_h = nn.Sequential(
            nn.Linear(
                node_attr_h
                + algebra.n_subspaces * hidden_features_x
                + in_features_h
                + hidden_features_h,
                hidden_features_h,
            ),
            nn.BatchNorm1d(hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_h),
        )

        if aggregation == "mean":
            self.aggregation = unsorted_segment_mean
        elif aggregation == "sum":
            self.aggregation = unsorted_segment_sum
        else:
            raise ValueError(f"Unknown aggregation {aggregation}")

        self.use_invariants_to_update = use_invariants_to_update
        self.residual = residual
        self.layer_type = layer_type
        self.out_features_x = out_features_x
        self.in_features_x = in_features_x
        self.in_features_h = in_features_h
        self.out_features_h = out_features_h

        self.psi_x = nn.Sequential(
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_x * self.algebra.n_subspaces),
        )

        self.chi_x = nn.Sequential(
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_x * self.algebra.n_subspaces),
        )

        self.aggregation = aggregation

    def reduce(self, input, segment_ids, num_segments):
        if self.aggregation == "mean":
            red = unsorted_segment_mean(input, segment_ids, num_segments=num_segments)
        elif self.aggregation == "sum":
            red = unsorted_segment_sum(input, segment_ids, num_segments=num_segments)
        else:
            raise ValueError(f"Invalid aggregation function {self.aggregation}.")
        return red

    def message_x(self, x_i, x_j, edge_attr_x=None):
        x_diff = x_i - x_j

        input = [x_i, x_j, x_diff]

        if edge_attr_x is not None:
            input.append(edge_attr_x)

        input = torch.cat(input, dim=1)
        return self.phi_x(input)

    def message_h(self, h_i, h_j, invariants_ij, edge_attr_h=None):
        input = [invariants_ij, h_i, h_j, h_i - h_j]
        if edge_attr_h is not None:
            input.append(edge_attr_h)

        input = torch.cat(input, dim=1)
        return self.phi_h(input)

    def update_x(self, x, x_red, node_attr_x):
        if node_attr_x is not None:
            input = torch.cat([x, x_red, node_attr_x], dim=1)
        else:
            input = torch.cat([x, x_red], dim=1)

        return self.theta_x(input)

    def update_h(self, h, h_red, invariants_i, node_attr_h):
        if node_attr_h is not None:
            input = torch.cat([h, h_red, invariants_i, node_attr_h], dim=1)
        else:
            input = torch.cat(
                [
                    h,
                    h_red,
                    invariants_i,
                ],
                dim=1,
            )

        return self.theta_h(input)

    def forward(self, h, x, edges, node_attr_h, node_attr_x, edge_attr_h, edge_attr_x):
        i, j = edges
        m_x = self.message_x(x[i], x[j], edge_attr_x)
        m_invariants = get_invariants(self.algebra, m_x).flatten(1)

        if h is not None:
            m_h = self.message_h(h[i], h[j], m_invariants, edge_attr_h)
        else:
            m_h = None

        if self.use_invariants_to_update:
            weights = self.psi_x(m_h).view(
                len(m_h), self.out_features_x, self.algebra.n_subspaces
            )
            weights = torch.repeat_interleave(weights, self.algebra.subspaces, dim=2)
            m_x = m_x * torch.sigmoid(weights)

        x_red = self.reduce(m_x.flatten(1), i, num_segments=x.size(0)).view(
            len(x), *m_x.shape[1:]
        )

        if m_h is not None:
            h_red = self.reduce(m_h, i, num_segments=h.size(0))
        else:
            h_red = None

        x_u = self.update_x(x, x_red, node_attr_x)
        u_invariants = get_invariants(self.algebra, x).flatten(1)

        if h_red is not None:
            h_u = self.update_h(h, h_red, u_invariants, node_attr_h)

        if self.use_invariants_to_update:
            weights = self.chi_x(h_u).view(
                len(h_u), self.out_features_x, self.algebra.n_subspaces
            )

            weights = torch.repeat_interleave(weights, self.algebra.subspaces, dim=2)
            x_u = x_u * torch.sigmoid(weights)

        if self.residual and self.in_features_h == self.out_features_h:
            h = h_u + h
        else:
            h = h_u

        if self.residual and self.in_features_x == self.out_features_x:
            x = x_u + x
        else:
            x = x_u

        return h, x


class LorentzCGGNN(nn.Module):
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    def __init__(
        self,
        in_features_h: int = 2,
        hidden_features_h: int = 72,
        in_features_x: int = 1,
        hidden_features_x: int = 8,
        decoder_features=64,
        n_class=2,
        n_layers=4,
        dropout=0.2,
        use_invariants_to_update=True,
        use_invariant_network=True,
        normalization_init=None,
        residual=False,
        aggregation="mean",
        layer_type="fc",
    ):
        super().__init__()

        if not use_invariant_network:
            in_features_h = 0
            hidden_features_h = 0

        self.in_features_h = in_features_h
        self.hidden_features_h = hidden_features_h
        self.in_features_x = in_features_x
        self.hidden_features_x = hidden_features_x
        self.use_invariant_network = use_invariant_network

        self.algebra = CliffordAlgebra((1.0, -1.0, -1.0, -1.0))
        self.n_layers = n_layers
        self.embedding_h = nn.Linear(in_features_h, hidden_features_h)
        self.embedding_x = MVLinear(
            self.algebra, in_features_x, hidden_features_x, subspaces=False
        )
        self.CGLs = nn.ModuleList(
            [
                CGLayer(
                    self.algebra,
                    hidden_features_x,
                    hidden_features_x,
                    hidden_features_x,
                    hidden_features_h,
                    hidden_features_h,
                    hidden_features_h,
                    use_invariants_to_update=use_invariants_to_update,
                    normalization_init=normalization_init,
                    residual=residual,
                    aggregation=aggregation,
                    layer_type=layer_type,
                    node_attr_h=2,
                    node_attr_x=1,
                    edge_attr_x=3,
                )
                for i in range(n_layers)
            ]
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(
                hidden_features_h + hidden_features_x * self.algebra.n_subspaces,
                decoder_features,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_features, n_class),
        )  # classification

        self.train_metrics = MetricCollection(
            {
                "loss": Loss(),
                "accuracy": Accuracy(),
            },
        )
        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
                "accuracy": Accuracy(),
                "lorentz": LorentzMetric(),
            },
        )

    def _forward(
        self,
        h,
        x,
        edge_attr_h,
        edge_attr_x,
        node_attr_h,
        node_attr_x,
        edges,
        node_mask,
        edge_mask,
        n_nodes,
    ):
        if not self.use_invariant_network:
            h = None

        if h is not None:
            h = self.embedding_h(h)
        if x is not None:
            x = self.embedding_x(x)

        for i in range(self.n_layers):
            h, x = self.CGLs[i](
                h,
                x,
                edges,
                node_attr_x=node_attr_x,
                node_attr_h=node_attr_h,
                edge_attr_x=edge_attr_x,
                edge_attr_h=edge_attr_h,
            )

        invariants = get_invariants(self.algebra, x).flatten(1)

        if h is not None:
            h = torch.cat([h, invariants], dim=1)
        else:
            h = invariants

        h = h * node_mask
        h = h.view(
            -1,
            n_nodes,
            self.hidden_features_h + self.hidden_features_x * self.algebra.n_subspaces,
        )
        h = torch.mean(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)

    def forward(self, data, batch_idx):
        batch_size, n_nodes, _ = data["Pmu"].size()
        atom_momentum = data["Pmu"].view(batch_size * n_nodes, -1).float()
        atom_mask = data["atom_mask"].view(batch_size * n_nodes, -1)
        edge_mask = data["edge_mask"].reshape(batch_size * n_nodes * n_nodes, -1)
        nodes = data["nodes"].view(batch_size * n_nodes, -1).float()
        nodes = psi(nodes)
        edges = [a for a in data["edges"]]
        label = data["is_signal"].long()

        i, j = edges
        relative_momentum = atom_momentum[i] - atom_momentum[j]

        node_attr = []
        node_attr.append(atom_momentum)

        if len(node_attr) > 0:
            node_attr_x = torch.stack(node_attr, dim=1)
        else:
            node_attr_x = None

        edge_attr_x = []
        edge_attr_x.append(relative_momentum)
        edge_attr_x.append(atom_momentum[i])
        edge_attr_x.append(atom_momentum[j])

        if len(edge_attr_x) > 0:
            edge_attr_x = torch.stack(edge_attr_x, dim=1)
        else:
            edge_attr_x = None

        node_attr_h = []
        node_attr_h.append(nodes)

        if len(node_attr_h) > 0:
            node_attr_h = torch.cat(node_attr_h, dim=1)
        else:
            node_attr_h = None

        x = atom_momentum[:, None]
        h = nodes

        x = self.algebra.embed_grade(x, 1)
        if edge_attr_x is not None:
            edge_attr_x = self.algebra.embed_grade(edge_attr_x, 1)
        if node_attr_x is not None:
            node_attr_x = self.algebra.embed_grade(node_attr_x, 1)

        pred = self._forward(
            h=h,
            x=x,
            edge_attr_x=edge_attr_x,
            edge_attr_h=None,
            node_attr_x=node_attr_x,
            node_attr_h=node_attr_h,
            edges=edges,
            node_mask=atom_mask,
            edge_mask=edge_mask,
            n_nodes=n_nodes,
        )

        predict = pred.max(1).indices
        correct = (predict == label).float()
        loss = self.loss_fn(pred, label)
        preds = pred.softmax(dim=-1)[..., 1]

        return (
            loss.mean(),
            {
                "loss": loss,
                "accuracy": correct,
                "lorentz": (preds, label),
            },
        )
