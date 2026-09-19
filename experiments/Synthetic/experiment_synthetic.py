"""
experiment_synthetic.py
======================

Main entry point for the synthetic twin-pair experiment (graph
classification: big-vs-small hyperedges with identical clique expansion).

The structural quantities (P^EN, P^EE, P^WE, chi^EE, chi^WE) come from the
shared `eompp.eo.rw_quantities_sparse` - the same implementation the node
experiments use - so there is one EO formula in the repository.  The
graph-classification harness (disjoint-union batching, per-graph pooling,
training loop) is specific to this experiment and lives here.

Run
---
    python experiment_synthetic.py                 # default config
    python experiment_synthetic.py --quick         # small/fast sanity check
"""

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from eompp.eo import (build_incidence_scipy, rw_quantities_sparse,
                      KERNELS, CHI_KERNELS)
from eompp.metrics import macro_f1
from data_synthetic import make_dataset, split_dataset
from models_synthetic import build_model, MODEL_SPECS


# --------------------------------------------------------------------------
# Configuration.
# --------------------------------------------------------------------------
class Config:
    def __init__(self):
        # dataset
        self.n_base_graphs = 300
        self.feature_mode = "constant"   # "constant" or "random"
        self.feat_dim = 16
        self.k_range = (2, 5)            # number of cliques per base graph
        self.size_range = (3, 6)         # clique size range
        self.overlap_max = 2             # max shared vertices between cliques
        self.max_card = None             # K := max_e |e| over the dataset
        self.data_seed = 0
        self.split_seed = 0

        # model / training
        self.hidden = 64
        self.n_layers = 2
        self.lr = 1e-2
        self.weight_decay = 5e-4
        self.max_epochs = 300
        self.patience = 40
        self.seeds = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        
        # output
        self.results_path = "results_synthetic.json"


# --------------------------------------------------------------------------
# Batching: a split -> one disjoint-union batch of torch tensors.
# --------------------------------------------------------------------------
def build_batch(examples, max_card):
    """Concatenate a list of examples into a single batched graph."""

    xs, eis, batch_idx, ys = [], [], [], []
    p_lists = {k: [] for k in KERNELS}
    chi_lists = {k: [] for k in CHI_KERNELS}
    offset = 0
    for gi, ex in enumerate(examples):
        n = ex["n_nodes"]
        B = build_incidence_scipy(n, ex["hyperedges"])
        edge_index, p, chi, _ = rw_quantities_sparse(B, max_card)
        xs.append(ex["x"])
        eis.append(edge_index + offset)
        for k in KERNELS:
            p_lists[k].append(p[k])
        for k in CHI_KERNELS:
            chi_lists[k].append(chi[k])
        batch_idx.extend([gi] * n)
        ys.append(ex["label"])
        offset += n

    x = torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32)
    edge_index = torch.tensor(np.concatenate(eis, axis=1), dtype=torch.long)
    batch = torch.tensor(batch_idx, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)

    in_deg = torch.zeros(x.size(0), dtype=torch.float32)
    in_deg = in_deg.index_add(
        0,
        edge_index[1],
        torch.ones(edge_index.size(1)),
    )

    out = dict(x=x, edge_index=edge_index, in_deg=in_deg, batch=batch, y=y,
               num_graphs=len(examples))
    for k in KERNELS:
        out["p_" + k.lower()] = torch.tensor(
            np.concatenate(p_lists[k]), dtype=torch.float32)
    for k in CHI_KERNELS:
        out["chi_" + k.lower()] = torch.tensor(
            np.concatenate(chi_lists[k], axis=0), dtype=torch.float32)
    return out


def shuffle_chi(batch, key, seed):
    """Return a copy of `batch` with the rows of `key` randomly permuted."""

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(batch[key].size(0), generator=g)
    new = dict(batch)
    new[key] = batch[key][perm]
    return new


# --------------------------------------------------------------------------
# Evaluation / training (graph classification, full-batch).
# --------------------------------------------------------------------------
def evaluate(model, batch):
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    pred = logits.argmax(dim=-1)
    acc = float((pred == batch["y"]).float().mean())
    f1 = macro_f1(batch["y"], pred, n_classes=2)
    return acc, f1


def train_one(model, train_b, val_b, cfg):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    # Selection is on (val accuracy, -val loss).  Accuracy alone saturates
    # early on separable tasks: once it reaches its maximum it can never
    # improve strictly again, so the first, still under-converged model to
    # reach that maximum would be kept for the rest of training.  The loss
    # tie-break keeps improving the checkpoint after accuracy saturates.
    best_key, best_state, wait = (-1.0, float("inf")), None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(train_b), train_b["y"])
        loss.backward()
        opt.step()

        val_acc, _ = evaluate(model, val_b)
        with torch.no_grad():
            model.eval()
            val_loss = float(F.cross_entropy(model(val_b), val_b["y"]))
        key = (val_acc, -val_loss)
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# --------------------------------------------------------------------------
# Full experiment.
# --------------------------------------------------------------------------
def run(cfg):
    print("=" * 72)
    print("Synthetic experiment :  big vs small hyperedges, identical")
    print("                        clique expansion")
    print("=" * 72)
    print(f"feature_mode = {cfg.feature_mode!r}")

    # data
    examples = make_dataset(
        n_base=cfg.n_base_graphs, seed=cfg.data_seed,
        feature_mode=cfg.feature_mode,
        feat_dim=cfg.feat_dim, k_range=cfg.k_range,
        size_range=cfg.size_range, overlap_max=cfg.overlap_max)
    train_ex, val_ex, test_ex = split_dataset(examples, seed=cfg.split_seed)

    print(f"examples            : {len(examples)} "
          f"({sum(e['label'] == 0 for e in examples)} big / "
          f"{sum(e['label'] == 1 for e in examples)} small)")
    print(f"split  train/val/test : {len(train_ex)}/{len(val_ex)}/{len(test_ex)}")
    print("twin clique expansions verified equal by construction "
          "(assert in make_dataset)")

    # K := max_e |e|.  The graphs are batched into one tensor, so the three
    # splits must share one resolution: the maximum over the whole dataset,
    # which is the canonical K applied to their disjoint union.
    max_card = cfg.max_card
    if max_card is None:
        max_card = max((len(he) for ex in examples for he in ex["hyperedges"]
                        if len(he) >= 2), default=2)
        print(f"cardinality resolution: K = max_e |e| = {max_card} "
              f"(chi dimension {max_card - 1})")
    cfg.max_card = max_card             # record the resolved value in cfg,
                                         # so it is dumped in results_path

    train_b = build_batch(train_ex, max_card)
    val_b = build_batch(val_ex, max_card)
    test_b = build_batch(test_ex, max_card)
    chi_dim = train_b["chi_ee"].size(1)

    # train every model over every seed
    results = {}
    for name, shuffle_key in MODEL_SPECS:
        accs, f1s = [], []
        for seed in cfg.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            tr_b, va_b, te_b = train_b, val_b, test_b
            if shuffle_key:                         # variants E, H
                tr_b = shuffle_chi(train_b, shuffle_key, seed)
                va_b = shuffle_chi(val_b, shuffle_key, seed)
                te_b = shuffle_chi(test_b, shuffle_key, seed)

            model = build_model(name, feat_dim=cfg.feat_dim, chi_dim=chi_dim,
                                hidden=cfg.hidden, n_layers=cfg.n_layers)
            model = train_one(model, tr_b, va_b, cfg)
            acc, f1 = evaluate(model, te_b)
            accs.append(acc)
            f1s.append(f1)

        results[name] = dict(
            acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
            f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)),
            acc_per_seed=accs)
        print(f"  done: {name:<24s}  "
              f"acc = {results[name]['acc_mean'] * 100:5.1f} "
              f"+/- {results[name]['acc_std'] * 100:.1f}")

    # results
    print()
    print("=" * 72)
    print(f"RESULTS  (test set, mean +/- std over {len(cfg.seeds)} seeds)")
    print("=" * 72)
    print(f"{'Model':<26s} {'Accuracy (%)':>16s} {'Macro-F1 (%)':>16s}")
    print("-" * 72)
    for name, _ in MODEL_SPECS:
        r = results[name]
        print(f"{name:<26s} "
              f"{r['acc_mean'] * 100:8.1f} +/- {r['acc_std'] * 100:4.1f} "
              f"{r['f1_mean'] * 100:8.1f} +/- {r['f1_std'] * 100:4.1f}")
    print("-" * 72)
    print("Expected pattern: models without a cardinality descriptor")
    print("(Clique-GCN/GIN, A, B, F) near 50%; chi-based models (C, D, G)")
    print("near 100%; shuffled chi (E, H) collapses back to ~50%,")
    print("confirming the descriptors carry real structure.")

    out = dict(config=vars(cfg), results=results)
    with open(cfg.results_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved -> {cfg.results_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="small/fast config for a sanity check")
    parser.add_argument("--feature-mode", choices=["constant", "random"])
    args = parser.parse_args()

    cfg = Config()
    if args.quick:
        cfg.n_base_graphs = 80
        cfg.max_epochs = 120
        cfg.seeds = (0, 1)
        cfg.results_path = "results_synthetic_quick.json"
    if args.feature_mode:
        cfg.feature_mode = args.feature_mode

    run(cfg)


if __name__ == "__main__":
    sys.exit(main())
