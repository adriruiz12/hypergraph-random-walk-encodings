"""
data_benchmark.py
================

Loading of the real co-citation hypergraph benchmarks (Cora, Citeseer)
and assembly of the random-walk message passing data.

The random-walk quantities (P^EN, P^EE, P^WE, chi^EE, chi^WE) and the
SciPy incidence builder live in the shared `eompp.eo`; this module only
handles the benchmark-specific concerns: reading the committed pickle
datasets, building the incidence matrix with SciPy and cross-checking it
against TopoNetX when the package is available, the train/val/test split,
and the top-level loader.

Data source
-----------
Standard co-citation hypergraphs from HyperGCN (Yadati et al., 2019),
reused by AllSet (Chien et al., 2022).  The pickle files are committed
under ./data/; see data/README.md for provenance and checksums.
"""

import os
import pickle
import warnings
import numpy as np
import scipy.sparse as sp

from eompp.eo import build_incidence_scipy, rw_quantities_sparse

# benign: triggered by the pickled dtype of the HyperGCN dumps, not our code
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

DATASETS = ("cora", "citeseer")
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# absolute path -> works regardless of where the script is launched from


def load_raw(dataset, root=None):
    """Load (hyperedges, features, labels) from the local data folder."""

    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    base = os.path.join(root or DATA_ROOT, dataset)
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"dataset folder not found: {base}\n"
            f"See data/README.md for how to obtain the files.")
    with open(os.path.join(base, "hypergraph.pickle"), "rb") as fh:
        hg = pickle.load(fh)
    with open(os.path.join(base, "features.pickle"), "rb") as fh:
        features = pickle.load(fh)
    with open(os.path.join(base, "labels.pickle"), "rb") as fh:
        labels = pickle.load(fh)

    hyperedges = [sorted(set(int(v) for v in nodes)) for nodes in hg.values()]
    features = sp.csr_matrix(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    return hyperedges, features, labels


# --------------------------------------------------------------------------
# Incidence matrix via TopoNetX (cross-checked against eompp's scipy builder).
# --------------------------------------------------------------------------
def build_incidence_toponetx(n_nodes, hyperedges):
    """Build the N x M incidence matrix with TopoNetX (TopoX suite).

    TopoNetX returns rows in node-insertion order; we reorder them to the
    canonical 0..N-1 indexing and pad rows for any node that appears in
    no hyperedge of size >= 2.
    """

    import toponetx as tnx

    chg = tnx.ColoredHyperGraph()
    for he in hyperedges:
        if len(he) >= 2:
            chg.add_cell(list(he), rank=1)

    raw = chg.incidence_matrix(0, 1).tocoo()
    node_order = [int(next(iter(fs))) for fs in chg.nodes]  # row i -> node id
    perm = np.asarray(node_order, dtype=np.int64)
    new_row = perm[raw.row]                                 # remap rows
    B = sp.coo_matrix((raw.data, (new_row, raw.col)),
                      shape=(n_nodes, raw.shape[1])).tocsr()
    B.data[:] = 1.0
    return B


def build_incidence(n_nodes, hyperedges, use_toponetx=True, verify=True):
    """Build B with SciPy; cross-check against TopoNetX when available.

    SciPy is the production builder: it has no optional dependency and a
    deterministic column order, so results do not depend on whether
    TopoNetX happens to be installed (P^X and chi^X are already invariant to
    column order, being sums over Ecal(v) or Ecal(v,u), but the incidence
    matrix itself is consumed directly by AllDeepSets, whose sparse-matmul
    aggregation is not). The returned matrix is always the SciPy one; the
    cross-check validates our row-reordering glue for the TopoNetX path
    (perm[raw.row]), not TopoNetX itself: chg.nodes iteration order matching
    incidence_matrix row order is an undocumented internal assumption that
    could silently break.
    """

    B = build_incidence_scipy(n_nodes, hyperedges)
    if not use_toponetx:
        return B
    try:
        B_toponetx = build_incidence_toponetx(n_nodes, hyperedges)
    except Exception as exc:                       # pragma: no cover
        print(f"[data] TopoNetX unavailable ({exc}); skipping cross-check")
        return B
    if verify:
        # Column order may differ between the two builders; compare the
        # column *multisets* (each column is a hyperedge's node set).
        def colset(M):
            M = M.tocsc()
            return sorted(tuple(M.indices[M.indptr[j]:M.indptr[j + 1]])
                          for j in range(M.shape[1]))
        assert colset(B) == colset(B_toponetx), \
            "TopoNetX and scipy incidence matrices disagree"
    return B


# --------------------------------------------------------------------------
# Splits.
# --------------------------------------------------------------------------
def make_split(labels, seed, fracs=(0.5, 0.25, 0.25)):
    """Random class-stratified proportional train/validation/test split."""

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train, val, test = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        a = int(fracs[0] * n)
        b = int((fracs[0] + fracs[1]) * n)
        train += idx[:a].tolist()
        val += idx[a:b].tolist()
        test += idx[b:].tolist()
    return (np.array(sorted(train)), np.array(sorted(val)),
            np.array(sorted(test)))


# --------------------------------------------------------------------------
# Top-level loader.
# --------------------------------------------------------------------------
def load_dataset(dataset, max_card=None, root=None, use_toponetx=True):
    """Load one dataset and precompute everything the models need."""

    hyperedges, features, labels = load_raw(dataset, root)
    n_nodes = features.shape[0]

    B = build_incidence(n_nodes, hyperedges, use_toponetx=use_toponetx)
    edge_index, p, chi, d_h = rw_quantities_sparse(B, max_card)

    return dict(
        name=dataset,
        x=features.toarray().astype(np.float32),
        y=labels,
        n_nodes=n_nodes,
        n_features=features.shape[1],
        n_classes=int(labels.max()) + 1,
        incidence=B,                       # scipy csr, for AllDeepSets
        edge_index=edge_index,             # clique-expansion / RW edges
        p=p,                               # dict EN/EE/WE -> [E]
        chi=chi,                           # dict EE/WE    -> [E, K-1]
        d_h=d_h,
        chi_dim=chi["EE"].shape[1],
    )


if __name__ == "__main__":
    for ds in DATASETS:
        d = load_dataset(ds)
        # Every kernel must be row-stochastic: weights into each node sum to 1.
        N = d["n_nodes"]
        active = d["d_h"] > 0
        errs = {}
        for k, v in d["p"].items():
            acc = np.zeros(N)
            np.add.at(acc, d["edge_index"][1], v)
            errs[k] = float(np.abs(acc[active] - 1.0).max())
        print(f"{ds:9s}: N={N:6d}  F={d['n_features']:5d}  "
              f"C={d['n_classes']}  edges={d['edge_index'].shape[1]:8d}  "
              f"chi_dim={d['chi_dim']}  "
              + "  ".join(f"|rowsum(P^{k})-1|<={e:.1e}" for k, e in errs.items()))
