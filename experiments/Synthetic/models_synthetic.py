"""
models_synthetic.py
==================

Graph-classification models for the synthetic twin-pair experiment.

All models share the same shape:

    x  --(input projection)-->  L message-passing layers  -->
    mean pooling over nodes  -->  MLP classifier  -->  2 logits

The message-passing layers (GCN, GIN, RWPattern) are imported from the
shared `eompp.layers`, and the ablation ladder from `eompp.node_models`;
this module only defines the graph-classification wrappers (pooling + MLP
head).  Unlike the
node-classification nets in `eompp.node_models`, these use no dropout and
no residual: the synthetic graphs are tiny and the task is separable, so
the simplest wrapper keeps the 50%/100% result clean.
"""

import torch.nn as nn
import torch.nn.functional as F

from eompp.layers import scatter_mean, mlp, GCNLayer, GINLayer, RWPatternLayer
from eompp.node_models import RW_SPECS


# --------------------------------------------------------------------------
# Graph baselines on the clique expansion.
# --------------------------------------------------------------------------
class CliqueGNN(nn.Module):
    """GCN or GIN stack + per-graph mean pooling + MLP head.

    Operates solely on the pairwise graph induced by the hyperedges;
    hyperedge cardinalities are invisible to this model.
    """

    def __init__(self, feat_dim, hidden=64, n_layers=2, n_classes=2, kind="gcn"):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden)
        layer_cls = GCNLayer if kind == "gcn" else GINLayer
        self.layers = nn.ModuleList(layer_cls(hidden) for _ in range(n_layers))
        self.head = mlp(hidden, hidden, n_classes)

    def forward(self, batch):
        h = F.relu(self.input_proj(batch["x"]))
        for layer in self.layers:
            h = F.relu(layer(h, batch["edge_index"], batch["in_deg"]))
        graph = scatter_mean(h, batch["batch"], batch["num_graphs"])
        return self.head(graph)


# --------------------------------------------------------------------------
# The proposed hypergraph-derived model.
# --------------------------------------------------------------------------
class RWPatternNet(nn.Module):
    """RWPattern layer stack + per-graph mean pooling + MLP head."""

    def __init__(self, feat_dim, chi_dim, hidden=64, n_layers=2,
                 n_classes=2, kernel="EE", chi_mode="EE"):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden)
        self.layers = nn.ModuleList(
            RWPatternLayer(hidden, chi_dim, kernel, chi_mode)
            for _ in range(n_layers))
        self.head = mlp(hidden, hidden, n_classes)
        self.p_key = "p_" + kernel.lower()
        self.chi_key = None if chi_mode is None else "chi_" + chi_mode.lower()

    def forward(self, batch):
        h = F.relu(self.input_proj(batch["x"]))
        chi = None if self.chi_key is None else batch[self.chi_key]
        for layer in self.layers:
            h = F.relu(layer(h, batch["edge_index"], batch[self.p_key], chi))
        graph = scatter_mean(h, batch["batch"], batch["num_graphs"])
        return self.head(graph)


# --------------------------------------------------------------------------
# Registry.
# --------------------------------------------------------------------------
def build_model(name, feat_dim, chi_dim, hidden=64, n_layers=2):
    """Instantiate a model by name (see MODEL_SPECS)."""
    if name == "Clique-GCN":
        return CliqueGNN(feat_dim, hidden, n_layers, kind="gcn")
    if name == "Clique-GIN":
        return CliqueGNN(feat_dim, hidden, n_layers, kind="gin")
    if name in RW_SPECS:
        kernel, chi_mode = RW_SPECS[name]
        return RWPatternNet(feat_dim=feat_dim, chi_dim=chi_dim, hidden=hidden,
                            n_layers=n_layers, kernel=kernel,
                            chi_mode=chi_mode)
    raise ValueError(f"unknown model name: {name}")


# (model name, chi key to shuffle or None).  The lettered block is the same
# ablation ladder as in eompp.node_models, without the node-only baselines.
MODEL_SPECS = [
    ("Clique-GCN",                 None),
    ("Clique-GIN",                 None),
    ("A: EN",                      None),
    ("B: EE",                      None),
    ("C: EN + chi^EE",             None),
    ("D: EE-Pattern",              None),
    ("E: EE-Pattern, shuffled chi", "chi_ee"),
    ("F: WE",                      None),
    ("G: WE-Pattern",              None),
    ("H: WE-Pattern, shuffled chi", "chi_we"),
]
