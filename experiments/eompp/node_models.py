"""
eompp.node_models
=================

The twelve node-classification models shared by the Benchmark and
Complementarity experiments.  Every model maps node features X to per-node
class logits:

    X --(input proj)--> L message-passing layers --> linear classifier

  * MLP            : ignores all structure (control baseline).
  * Clique-GCN     : GCN on the clique expansion (graph baseline).
  * Clique-GIN     : GIN on the clique expansion (graph baseline).
  * AllDeepSets    : in-house mean-normalized Deep Sets-style baseline.
  * RWPatternNet   : the proposed model; (kernel, chi_mode) select the
                     paradigm and the ablation variant

                         A: EN / -        E: EE / EE, shuffled
                         B: EE / -        F: WE / -
                         C: EN / EE       G: WE / WE
                         D: EE / EE       H: WE / WE, shuffled

                     A-E reproduce the ablation ladder of the submitted
                     thesis (A = uniform baseline, D = EO-Pattern);
                     F-H add the weighted-edges paradigm.  Shuffling is
                     handled in eompp.node_training.shuffled_chi.

All models share dropout and a fixed hidden width for a fair comparison.
"""

import torch.nn as nn
import torch.nn.functional as F

from eompp.layers import (GCNLayer, GINLayer, AllDeepSetsLayer,
                          RWPatternLayer)


# --------------------------------------------------------------------------
# Control baseline: a plain MLP.
# --------------------------------------------------------------------------
class MLPNet(nn.Module):
    """Node classifier with no structural information at all."""

    def __init__(self, n_features, hidden, n_classes, n_layers=2, dropout=0.5):
        super().__init__()
        dims = [n_features] + [hidden] * n_layers
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers))
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, batch):
        h = batch["x"]
        for layer in self.layers:
            h = F.dropout(F.relu(layer(h)), self.dropout, self.training)
        return self.head(h)


# --------------------------------------------------------------------------
# Graph baselines on the clique expansion.
# --------------------------------------------------------------------------
class CliqueGNN(nn.Module):
    """GCN or GIN layer stack on the clique expansion + linear classifier.

    Operates solely on the pairwise graph induced by the hyperedges;
    hyperedge cardinalities are invisible to this model.
    """

    def __init__(self, n_features, hidden, n_classes, n_layers=2,
                 dropout=0.5, kind="gcn"):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        layer_cls = GCNLayer if kind == "gcn" else GINLayer
        self.layers = nn.ModuleList(layer_cls(hidden) for _ in range(n_layers))
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, batch):
        h = F.dropout(F.relu(self.input_proj(batch["x"])),
                      self.dropout, self.training)
        for layer in self.layers:
           out = F.relu(layer(h, batch["edge_index"], batch["deg"]))
           h = F.dropout(out + h, self.dropout, self.training)   # residual
        return self.head(h)
    

# --------------------------------------------------------------------------
# In-house mean-normalized Deep Sets-style incidence baseline.
# --------------------------------------------------------------------------
class AllDeepSetsNet(nn.Module):
    """Mean-normalized Deep Sets-style layer stack and linear classifier."""

    def __init__(self, n_features, hidden, n_classes, n_layers=2, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        self.layers = nn.ModuleList(
            AllDeepSetsLayer(hidden, dropout) for _ in range(n_layers))
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, batch):
        h = F.dropout(F.relu(self.input_proj(batch["x"])),
                      self.dropout, self.training)
        for layer in self.layers:
            out = layer(h, batch["incidence"], batch["incidence_t"],
                        batch["he_deg"], batch["node_deg_hg"])
            h = F.dropout(F.relu(out) + h, self.dropout, self.training)  # residual
        return self.head(h)


# --------------------------------------------------------------------------
# The proposed model: a random-walk-induced layer stack.
# --------------------------------------------------------------------------
class RWPatternNet(nn.Module):
    """RWPattern layer stack + linear classifier.

    `kernel` and `chi_mode` are fixed at construction; the corresponding
    tensors are read from the batch under the keys "p_<kernel>" and
    "chi_<chi_mode>" (lower case), so a single net class covers all
    paradigms without duplicating the forward pass.
    """

    def __init__(self, n_features, chi_dim, hidden, n_classes, n_layers=2,
                 dropout=0.5, kernel="EE", chi_mode="EE"):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        self.layers = nn.ModuleList(
            RWPatternLayer(hidden, chi_dim, kernel, chi_mode)
            for _ in range(n_layers))
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = dropout
        self.p_key = "p_" + kernel.lower()
        self.chi_key = None if chi_mode is None else "chi_" + chi_mode.lower()

    def forward(self, batch):
        h = F.dropout(F.relu(self.input_proj(batch["x"])),
                      self.dropout, self.training)
        chi = None if self.chi_key is None else batch[self.chi_key]
        for layer in self.layers:
            out = layer(h, batch["edge_index"], batch[self.p_key], chi)
            h = F.dropout(F.relu(out) + h, self.dropout, self.training)  # residual
        return self.head(h)


# --------------------------------------------------------------------------
# Registry.
# --------------------------------------------------------------------------
# name -> (kernel, chi_mode).  Shared with the synthetic experiment so that
# the ablation ladder is defined exactly once.
RW_SPECS = {
    "A: EN":                       ("EN", None),
    "B: EE":                       ("EE", None),
    "C: EN + chi^EE":              ("EN", "EE"),
    "D: EE-Pattern":               ("EE", "EE"),
    "E: EE-Pattern, shuffled chi": ("EE", "EE"),
    "F: WE":                       ("WE", None),
    "G: WE-Pattern":               ("WE", "WE"),
    "H: WE-Pattern, shuffled chi": ("WE", "WE"),
}


def build_model(name, n_features, chi_dim, n_classes,
                hidden=128, n_layers=2, dropout=0.5):
    """Instantiate a model by name (see MODEL_SPECS)."""
    common = dict(n_features=n_features, hidden=hidden,
                  n_classes=n_classes, n_layers=n_layers, dropout=dropout)
    if name == "MLP":
        return MLPNet(**common)
    if name == "Clique-GCN":
        return CliqueGNN(kind="gcn", **common)
    if name == "Clique-GIN":
        return CliqueGNN(kind="gin", **common)
    if name == "AllDeepSets":
        return AllDeepSetsNet(**common)
    if name in RW_SPECS:
        kernel, chi_mode = RW_SPECS[name]
        return RWPatternNet(chi_dim=chi_dim, kernel=kernel,
                            chi_mode=chi_mode, **common)
    raise ValueError(f"unknown model name: {name}")


# (model name, chi key to shuffle or None).  First block is the main
# comparison; the lettered block is the ablation study.
MODEL_SPECS = [
    ("MLP",                        None),
    ("Clique-GCN",                 None),
    ("Clique-GIN",                 None),
    ("AllDeepSets",                None),
    ("A: EN",                      None),
    ("B: EE",                      None),
    ("C: EN + chi^EE",             None),
    ("D: EE-Pattern",              None),
    ("E: EE-Pattern, shuffled chi", "chi_ee"),
    ("F: WE",                      None),
    ("G: WE-Pattern",              None),
    ("H: WE-Pattern, shuffled chi", "chi_we"),
]
