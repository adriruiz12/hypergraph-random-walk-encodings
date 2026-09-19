"""
eompp.gadgets
=============

Shared driver for the controlled node-classification gadgets.

A gadget is a synthetic hypergraph in which one designated neighbour of each
target vertex carries the target's label and all other neighbours are decoys
that match the informative one on some, but not all, of the random-walk pair
statistics.  Which statistics are matched is what each gadget varies; the
training protocol is identical, so this module holds it once:

  * `split_targets`  : class-stratified split over the target vertices only.
  * `run_gadget`     : train every model in MODEL_SPECS over a list of seeds,
                       with resumable checkpointing.

The gadget-specific code (generator and signature check) lives with each
experiment.
"""

import types

import numpy as np
import torch

from eompp.node_models import build_model, MODEL_SPECS
from eompp.node_training import to_tensors, train_one, evaluate, shuffled_chi


# --------------------------------------------------------------------------
# Split over target nodes only.
# --------------------------------------------------------------------------
def split_targets(y, targets, seed, fracs=(0.5, 0.25, 0.25)):
    """Class-stratified train/val/test split restricted to `targets`."""

    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y[targets]):
        idx = targets[y[targets] == c]
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
        tr += idx[:a].tolist()
        va += idx[a:b].tolist()
        te += idx[b:].tolist()
    return np.array(tr), np.array(va), np.array(te)


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------
def run_gadget(data, seeds, hidden, n_layers, dropout, lr, weight_decay,
               max_epochs, patience, prev=None, save_cb=None):
    """Train every model in MODEL_SPECS on one fixed gadget hypergraph.

    Only the split and the weight initialization vary across seeds; the
    structure is fixed, so the comparison isolates what each model can read
    from that structure.
    """

    device = torch.device("cpu")
    cfg = types.SimpleNamespace(lr=lr, weight_decay=weight_decay,
                                max_epochs=max_epochs, patience=patience)
    batch = to_tensors(data, device)
    targets = data["targets"]
    n_classes = data["n_classes"]
    print(f"nodes={data['n_nodes']}  targets={len(targets)}  "
          f"hyperedges={data['incidence'].shape[1]}  "
          f"edges={data['edge_index'].shape[1]}  chi_dim={data['chi_dim']}\n")

    print(f"{'Model':<30s} {'Accuracy (%)':>15s}")
    print("-" * 48)
    results = dict(prev) if prev else {}
    results["_meta"] = dict(max_card=data["chi_dim"] + 1)  # resolved K
    for name, shuffle_key in MODEL_SPECS:
        cached = results.get(name, {})
        accs = list(cached.get("acc_per_seed", []))
        if len(accs) >= len(seeds):
            print(f"  skip: {name}")
            continue
        for seed in seeds[len(accs):]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            tr, va, te = split_targets(data["y"], targets, seed)

            def mask(idx):
                return torch.zeros(data["n_nodes"], dtype=torch.bool,
                                   device=device).index_fill_(
                    0, torch.tensor(idx, device=device), True)

            b = (shuffled_chi(batch, shuffle_key, seed)
                 if shuffle_key else batch)
            model = build_model(
                name, n_features=data["n_features"], chi_dim=data["chi_dim"],
                n_classes=n_classes, hidden=hidden, n_layers=n_layers,
                dropout=dropout).to(device)
            model = train_one(model, b, mask(tr), mask(va), n_classes, cfg)
            acc, _ = evaluate(model, b, mask(te), n_classes)
            accs.append(float(acc))
            results[name] = dict(acc_mean=float(np.mean(accs)),
                                 acc_std=float(np.std(accs)),
                                 acc_per_seed=accs)
            if save_cb:
                save_cb(results)
        print(f"{name:<30s} {np.mean(accs) * 100:7.1f} "
              f"+/- {np.std(accs) * 100:4.1f}")
    print("-" * 48)
    return results
