"""
pqr.py
======

Experiment 3 - Complementarity.  A controlled task where BOTH hypergraph-
native ingredients of the EO-Pattern layer (the random-walk weight P^EE and
the cardinality descriptor chi) are simultaneously *load-bearing*.

Motivation
----------
In the twin-pair synthetic, with constant node features any row-stochastic
weight (uniform mean or P^EE) is inert, so only chi discriminates (P^EE itself
differs between the twins but is unusable on featureless nodes).  In the real 
co-citation benchmark, structured chi provides no observable benefit, while 
P^EE yields at most a small gain over EO-A. Neither experiment demonstrates
a clear empirical benefit from combining both quantities.

This experiment fixes that with the P/Q/R gadget.  For each target node v:

    P-type neighbor : one size-2 hyperedge  {v, uP}             -> Z = 1,   chi = e2
    Q-type neighbor : two size-3 hyperedges {v,uQ,x1},{v,uQ,x2} -> Z = 1,   chi = e3
    R-type neighbor : one size-3 hyperedge  {v, uR, x3}         -> Z = 1/2, chi = e3


  * chi separates P from {Q,R} but NOT Q from R (normalisation erases multiplicity).
  * P^EE separates R from {P,Q} but NOT P from Q because the scalar connection mass
  Z does not retain its cardinality composition.
  * Only (chi, P^EE) jointly single out Q = (e3, Z=1).

The Q-neighbour carries the true class of v in its feature; P/R neighbours are
decoys with random classes; v and the fillers x_i are neutral.  Recovering y_v is
easiest by reading Q; weaker models retain a diluted label signal through averaging,
which explains their non-chance floor.

The shared core (eompp) provides rw_quantities_sparse, the twelve models and the
train/eval routines; the only experiment-specific code here is the P/Q/R
generator and the target-only split.

Run
---
    python pqr.py                                 # all twelve models, 10 seeds
    python pqr.py --n-gadgets 450 --m-decoys 4 --n-seeds 10 --n-layers 1
"""

import argparse
import json
import os
import types

import numpy as np
import torch

from eompp.eo import build_incidence_scipy, rw_quantities_sparse
from eompp.node_models import build_model, MODEL_SPECS
from eompp.node_training import to_tensors, train_one, evaluate, shuffled_chi


# --------------------------------------------------------------------------
# Generator.
# --------------------------------------------------------------------------
def make_pqr_dataset(n_gadgets=800, n_classes=2, m_decoys=2, seed=0,
                     max_card=None, noise=0.01):
    """Build a P/Q/R synthetic hypergraph as a load_dataset-style dict."""

    rng = np.random.default_rng(seed)
    hyperedges = []
    feat_class = {}                 # node -> class index, or -1 for neutral
    label = {}                      # target node -> class
    nid = 0

    def new():
        """Node counter"""
    
        nonlocal nid
        nid += 1
        return nid - 1

    for _ in range(n_gadgets):
        y = int(rng.integers(n_classes))
        v = new()
        label[v] = y
        feat_class[v] = -1

        # Q: the unique (e3, Z=1) signature -> two size-3 hyperedges
        uQ = new()
        feat_class[uQ] = y                       # tells the truth
        for _ in range(2):
            x = new()
            feat_class[x] = -1
            hyperedges.append([v, uQ, x])

        # P-type decoys: (e2, Z=1) -> one size-2 hyperedge each
        for _ in range(m_decoys):
            uP = new()
            feat_class[uP] = int(rng.integers(n_classes))
            hyperedges.append([v, uP])

        # R-type decoys: (e3, Z=1/2) -> one size-3 hyperedge each
        for _ in range(m_decoys):
            uR = new()
            feat_class[uR] = int(rng.integers(n_classes))
            x = new()
            feat_class[x] = -1
            hyperedges.append([v, uR, x])

    N = nid
    B = build_incidence_scipy(N, hyperedges)
    edge_index, p, chi, d_h = rw_quantities_sparse(B, max_card)

    X = np.zeros((N, n_classes), dtype=np.float32)
    for node, c in feat_class.items():
        if c >= 0:
            X[node, c] = 1.0
    X += noise * rng.standard_normal(X.shape).astype(np.float32)

    y = np.full(N, -1, dtype=np.int64)
    for v, c in label.items():
        y[v] = c
    targets = np.array(sorted(label.keys()), dtype=np.int64)

    return dict(
        name="pqr", x=X, y=y, n_nodes=N, n_features=n_classes,
        n_classes=n_classes, incidence=B, edge_index=edge_index,
        p=p, chi=chi, d_h=d_h, chi_dim=chi["EE"].shape[1], targets=targets)


# --------------------------------------------------------------------------
# Split over target nodes only.
# --------------------------------------------------------------------------
def split_targets(y, targets, seed, fracs=(0.5, 0.25, 0.25)):
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
# Sanity check: verify the P/Q/R signatures are exactly as claimed.
# --------------------------------------------------------------------------
def check_signatures(data):
    """Print the (chi-argmax, Z, c) signatures of the first target's
    neighbours, to confirm P=(e2,1,1), Q=(e3,1,2), R=(e3,.5,1).

    The third coordinate is the co-occurrence multiplicity c_vu seen by the
    WE kernel; it is reported because it separates Q on its own, whereas
    (chi^EE, Z) separates Q only jointly.
    """

    src, tgt = data["edge_index"]
    v = int(data["targets"][0])
    mask = tgt == v
    sigs = {}
    Z = data["p"]["EE"][mask] * data["d_h"][v]      # recover Z^EE
    card = data["chi"]["EE"][mask].argmax(axis=1) + 2
    B = data["incidence"]
    c_vu = np.asarray((B[v] @ B.T).todense()).ravel()[src[mask]]  # |E(v,u)|
    for a, z, c in zip(card, np.round(Z, 3), c_vu):
        key = (int(a), float(z), int(c))
        sigs[key] = sigs.get(key, 0) + 1
    print(f"[check] target v={v}  d_H={data['d_h'][v]:.0f}  "
          f"signatures (card, Z^EE, c_vu) -> count: {sigs}")


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------
def run(n_gadgets, n_classes, m_decoys, seeds, hidden, n_layers, dropout,
        lr, weight_decay, max_epochs, patience, prev=None, save_cb=None):
    device = torch.device("cpu")
    cfg = types.SimpleNamespace(lr=lr, weight_decay=weight_decay,
                                max_epochs=max_epochs, patience=patience)

    # one fixed structure; only the split + init vary across seeds
    data = make_pqr_dataset(n_gadgets=n_gadgets, n_classes=n_classes,
                            m_decoys=m_decoys, seed=12345)
    check_signatures(data)
    batch = to_tensors(data, device)
    targets = data["targets"]
    print(f"nodes={data['n_nodes']}  targets={len(targets)}  "
          f"hyperedges={data['incidence'].shape[1]}  "
          f"edges={data['edge_index'].shape[1]}  chi_dim={data['chi_dim']}\n")

    print(f"{'Model':<28s} {'Accuracy (%)':>15s}")
    print("-" * 46)
    results = dict(prev) if prev else {}
    results["_meta"] = dict(max_card=data["chi_dim"] + 1)  # resolved K
    for name, shuffle_key in MODEL_SPECS:
        cached = results.get(name, {})
        accs = list(cached.get("acc_per_seed", []))
        done = len(accs)
        if done >= len(seeds):
            print(f"  skip: {name}")
            continue
        for seed in seeds[done:]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            tr, va, te = split_targets(data["y"], targets, seed)
            mk = lambda idx: torch.zeros(data["n_nodes"], dtype=torch.bool,
                                         device=device).index_fill_(
                0, torch.tensor(idx, device=device), True)
            train_mask, val_mask, test_mask = mk(tr), mk(va), mk(te)

            b = (shuffled_chi(batch, shuffle_key, seed)
                 if shuffle_key else batch)
            model = build_model(
                name, n_features=data["n_features"], chi_dim=data["chi_dim"],
                n_classes=n_classes, hidden=hidden, n_layers=n_layers,
                dropout=dropout).to(device)
            model = train_one(model, b, train_mask, val_mask, n_classes, cfg)
            acc, _ = evaluate(model, b, test_mask, n_classes)
            accs.append(acc)
            results[name] = dict(acc_mean=float(np.mean(accs)),
                                 acc_std=float(np.std(accs)),
                                 acc_per_seed=[float(a) for a in accs])
            if save_cb:
                save_cb(results)
        results[name] = dict(acc_mean=float(np.mean(accs)),
                             acc_std=float(np.std(accs)),
                             acc_per_seed=[float(a) for a in accs])
        print(f"{name:<28s} {np.mean(accs) * 100:7.1f} "
              f"+/- {np.std(accs) * 100:4.1f}")
    print("-" * 46)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gadgets", type=int, default=450)
    ap.add_argument("--n-classes", type=int, default=2)
    ap.add_argument("--m-decoys", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--out", type=str, default="results_complementarity.json")
    args = ap.parse_args()

    prev = {}
    if args.out and os.path.exists(args.out):
        with open(args.out) as fh:
            prev = json.load(fh).get("results", {})

    config = dict(n_gadgets=args.n_gadgets, n_classes=args.n_classes,
                  m_decoys=args.m_decoys, n_seeds=args.n_seeds,
                  hidden=args.hidden, n_layers=args.n_layers,
                  dropout=args.dropout, lr=args.lr,
                  weight_decay=args.weight_decay,
                  max_epochs=args.max_epochs, patience=args.patience)

    def save_cb(results):
        with open(args.out, "w") as fh:
            json.dump(dict(config=config, results=results), fh, indent=2)

    results = run(
        n_gadgets=args.n_gadgets, n_classes=args.n_classes,
        m_decoys=args.m_decoys, seeds=tuple(range(args.n_seeds)),
        hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout,
        lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, patience=args.patience,
        prev=prev, save_cb=save_cb if args.out else None)

    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()
