"""
pqr_we.py
=========

Experiment 3b - Complementarity, WE factorization.  The mirror image of
`pqr.py`.

Motivation
----------
The P/Q/R gadget of `pqr.py` is built against the EE factorization: the
informative neighbour Q is identified by the joint signature (P^EE, chi^EE)
and by neither alone.  It is NOT neutral with respect to WE, because Q is the
only neighbour there with co-occurrence multiplicity c_vu = 2, and P^WE is
exactly the kernel that retains multiplicity.  A gadget designed against one
factorization therefore says nothing about the other.

This experiment builds the symmetric construction, in which the roles are
exchanged.  For each target node v:

    P-type neighbour : two size-4 hyperedges  -> c_vu = 2,  chi^WE = e4
    Q-type neighbour : two size-3 hyperedges  -> c_vu = 2,  chi^WE = e3
    R-type neighbour : one size-3 hyperedge   -> c_vu = 1,  chi^WE = e3

  * P^WE separates R from {P,Q} (multiplicity) but NOT P from Q.
  * chi^WE separates P from {Q,R} (cardinality) but NOT Q from R.
  * Only the joint signature (c_vu = 2, e3) singles out Q.

The EE quantities behave in the mirror-image way.  Their pair masses are

    Z^EE_P = 2/3,     Z^EE_Q = 1,     Z^EE_R = 1/2,

all three distinct, so P^EE alone identifies Q, exactly as P^WE alone
identifies Q in the EE gadget.  Taken together the two gadgets show that
complementarity is a property of the chosen factorization and not of the
hypergraph information: whichever paradigm a gadget is built against needs
both of its ingredients, while the other paradigm resolves the same task with
its kernel alone.

Run
---
    python pqr_we.py
    python pqr_we.py --n-gadgets 450 --m-decoys 4 --n-seeds 10
"""

import argparse
import json
import os

import numpy as np

from eompp.eo import build_incidence_scipy, rw_quantities_sparse
from eompp.gadgets import run_gadget


# --------------------------------------------------------------------------
# Generator.
# --------------------------------------------------------------------------
def make_pqr_we_dataset(n_gadgets=450, n_classes=2, m_decoys=4, seed=0,
                        max_card=None, noise=0.01):
    """Build the WE-targeted P/Q/R gadget as a load_dataset-style dict."""

    rng = np.random.default_rng(seed)
    hyperedges = []
    feat_class = {}
    label = {}
    nid = 0

    def new():
        """Node counter"""

        nonlocal nid
        nid += 1
        return nid - 1

    def fillers(k):
        out = []
        for _ in range(k):
            x = new()
            feat_class[x] = -1
            out.append(x)
        return out

    for _ in range(n_gadgets):
        y = int(rng.integers(n_classes))
        v = new()
        label[v] = y
        feat_class[v] = -1

        # Q: the unique (c_vu = 2, e3) signature -> two size-3 hyperedges
        uQ = new()
        feat_class[uQ] = y                       # tells the truth
        for _ in range(2):
            hyperedges.append([v, uQ] + fillers(1))

        # P-type decoys: (c_vu = 2, e4) -> two size-4 hyperedges each
        for _ in range(m_decoys):
            uP = new()
            feat_class[uP] = int(rng.integers(n_classes))
            for _ in range(2):
                hyperedges.append([v, uP] + fillers(2))

        # R-type decoys: (c_vu = 1, e3) -> one size-3 hyperedge each
        for _ in range(m_decoys):
            uR = new()
            feat_class[uR] = int(rng.integers(n_classes))
            hyperedges.append([v, uR] + fillers(1))

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
        name="pqr_we", x=X, y=y, n_nodes=N, n_features=n_classes,
        n_classes=n_classes, incidence=B, edge_index=edge_index,
        p=p, chi=chi, d_h=d_h, chi_dim=chi["EE"].shape[1], targets=targets)


# --------------------------------------------------------------------------
# Sanity check.
# --------------------------------------------------------------------------
def check_signatures(data):
    """Print the (chi^WE-argmax, c_vu, Z^EE) signatures of one target's
    neighbourhood, to confirm P=(e4,2,2/3), Q=(e3,2,1), R=(e3,1,1/2)."""

    src, tgt = data["edge_index"]
    v = int(data["targets"][0])
    mask = tgt == v
    B = data["incidence"]
    c_vu = np.asarray((B[v] @ B.T).todense()).ravel()[src[mask]]
    card = data["chi"]["WE"][mask].argmax(axis=1) + 2
    Z = data["p"]["EE"][mask] * data["d_h"][v]
    sigs = {}
    for a, c, z in zip(card, c_vu, np.round(Z, 3)):
        key = (int(a), int(c), float(z))
        sigs[key] = sigs.get(key, 0) + 1
    print(f"[check] target v={v}  d_H={data['d_h'][v]:.0f}  "
          f"signatures (card, c_vu, Z^EE) -> count: {sigs}")


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------
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
    ap.add_argument("--out", type=str, default="results_complementarity_we.json")
    args = ap.parse_args()

    prev = {}
    if args.out and os.path.exists(args.out):
        with open(args.out) as fh:
            prev = json.load(fh).get("results", {})

    config = vars(args)

    def save_cb(results):
        with open(args.out, "w") as fh:
            json.dump(dict(config=config, results=results), fh, indent=2)

    data = make_pqr_we_dataset(
        n_gadgets=args.n_gadgets, n_classes=args.n_classes,
        m_decoys=args.m_decoys, seed=12345)
    check_signatures(data)

    run_gadget(data, seeds=tuple(range(args.n_seeds)), hidden=args.hidden,
               n_layers=args.n_layers, dropout=args.dropout, lr=args.lr,
               weight_decay=args.weight_decay, max_epochs=args.max_epochs,
               patience=args.patience, prev=prev,
               save_cb=save_cb if args.out else None)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
