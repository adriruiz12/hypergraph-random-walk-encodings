"""
multiplicity.py
===============

Experiment 4 - Multiplicity.  The mirror image of the cardinality twin
experiment: a controlled task in which the discriminating structure is
co-occurrence MULTIPLICITY at fixed hyperedge cardinality.

Motivation
----------
The twin experiment of `Synthetic/` varies hyperedge cardinality at fixed
adjacency, and shows that the descriptors chi^EE and chi^WE recover structure
the clique expansion destroys while no kernel can (with constant features any
row-stochastic weight is inert).  It says nothing about multiplicity.

This experiment varies multiplicity at fixed cardinality.  Every hyperedge has
size 3, so the hypergraph is 3-uniform and, by the structural-inertness
proposition,

    chi^EE_vu = chi^WE_vu = c_3     for every adjacent pair,

i.e. both descriptors are provably constant and carry no information at all.
For the same reason the two kernels coincide: on an r-uniform hypergraph

    Z^EE_vu = c_vu/(r-1),   s_H(v) = (r-1) d_H(v)
    =>  P^EE(v,u) = c_vu / ((r-1) d_H(v)) = P^WE(v,u),

so variants B and F, and likewise D and G, must return identical numbers.
That identity is a built-in correctness check, not a coincidence.

Construction
------------
For each target node v:

    Q-type neighbour : two size-3 hyperedges {v,uQ,x1},{v,uQ,x2}  -> c_vu = 2
    m decoys         : one size-3 hyperedge  {v,uD,x}             -> c_vu = 1

The Q neighbour carries the true class of v in its feature; decoys carry
random classes; v and the fillers x are neutral.  All neighbours are
adjacent to v and share the same hyperedge cardinality, so the informative
one is identifiable only from how many hyperedges it shares with v.  This is
exactly the information EN discards and both hyperedge-mediated kernels keep.

Run
---
    python multiplicity.py
    python multiplicity.py --n-gadgets 450 --m-decoys 4 --n-seeds 10
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
def make_multiplicity_dataset(n_gadgets=450, n_classes=2, m_decoys=4, seed=0,
                              max_card=None, noise=0.01):
    """Build a 3-uniform multiplicity gadget as a load_dataset-style dict."""

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

        # Q: the unique c_vu = 2 signature -> two size-3 hyperedges
        uQ = new()
        feat_class[uQ] = y                       # tells the truth
        for _ in range(2):
            x = new()
            feat_class[x] = -1
            hyperedges.append([v, uQ, x])

        # decoys: c_vu = 1 -> one size-3 hyperedge each
        for _ in range(m_decoys):
            uD = new()
            feat_class[uD] = int(rng.integers(n_classes))
            x = new()
            feat_class[x] = -1
            hyperedges.append([v, uD, x])

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
        name="multiplicity", x=X, y=y, n_nodes=N, n_features=n_classes,
        n_classes=n_classes, incidence=B, edge_index=edge_index,
        p=p, chi=chi, d_h=d_h, chi_dim=chi["EE"].shape[1], targets=targets)


# --------------------------------------------------------------------------
# Sanity check: the descriptors must be constant and the kernels must agree.
# --------------------------------------------------------------------------
def check_signatures(data):
    """Verify 3-uniformity implies constant chi and P^EE == P^WE."""

    src, tgt = data["edge_index"]
    chi_ee, chi_we = data["chi"]["EE"], data["chi"]["WE"]
    const_ee = bool(np.allclose(chi_ee, chi_ee[0]))
    const_we = bool(np.allclose(chi_we, chi_we[0]))
    same_kernel = bool(np.allclose(data["p"]["EE"], data["p"]["WE"]))

    v = int(data["targets"][0])
    mask = tgt == v
    B = data["incidence"]
    c_vu = np.asarray((B[v] @ B.T).todense()).ravel()[src[mask]]
    card = chi_ee[mask].argmax(axis=1) + 2
    sigs = {}
    for a, c in zip(card, c_vu):
        sigs[(int(a), int(c))] = sigs.get((int(a), int(c)), 0) + 1

    print(f"[check] chi^EE constant: {const_ee}   chi^WE constant: {const_we}"
          f"   P^EE == P^WE: {same_kernel}")
    print(f"[check] target v={v}  d_H={data['d_h'][v]:.0f}  "
          f"signatures (card, c_vu) -> count: {sigs}")
    assert const_ee and const_we and same_kernel, \
        "3-uniformity broken: the experiment would not be diagnostic"


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
    ap.add_argument("--out", type=str, default="results_multiplicity.json")
    args = ap.parse_args()

    prev = {}
    if args.out and os.path.exists(args.out):
        with open(args.out) as fh:
            prev = json.load(fh).get("results", {})

    config = vars(args)

    def save_cb(results):
        with open(args.out, "w") as fh:
            json.dump(dict(config=config, results=results), fh, indent=2)

    data = make_multiplicity_dataset(
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
