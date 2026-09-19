"""
eompp.rw
========

Hypergraph helpers and the random-walk structural quantities shared by all
three experiments.  This is the single source of truth for the hypergraph-
derived quantities used below.  Three vertex-level random walks on a
hypergraph H = (V, E) induce three transition kernels:

  * P^EN(v, u) = 1 / |N_H(v)|
        equal-nodes: pick a distinct neighbour uniformly.  This is the walk
        on the unweighted clique expansion; it retains only adjacency.

  * P^EE(v, u) = Z^EE_vu / d_H(v),   Z^EE_vu = sum_{e in E(v,u)} 1/(|e|-1)
        equal-edges: pick an incident hyperedge uniformly, then a vertex
        inside it uniformly.

  * P^WE(v, u) = c_vu / s_H(v),      c_vu = |E(v,u)|,
                                     s_H(v) = sum_{e in E(v)} (|e|-1)
        weighted-edges: pick an incident hyperedge with probability
        proportional to |e|-1, then a vertex inside it uniformly.  The
        (|e|-1) factors cancel, so only co-occurrence multiplicity survives.

Each of the two hyperedge-mediated walks induces a cardinality descriptor,
namely the binned conditional distribution of |e| given the transition:

  * chi^EE_vu : 1/(|e|-1)-weighted one-hot histogram of the cardinalities
        of the hyperedges shared by v and u, normalized by Z^EE_vu.
  * chi^WE_vu : equally weighted one-hot histogram of the same
        cardinalities, normalized by c_vu.

EN mediates no hyperedge choice, so it induces no descriptor.

The cardinality resolution is K := max_{e in E} |e|, so that cardinalities
2, ..., K each get their own bin (K-1 values for K-1 dimensions) and the
final bin never merges two distinct observed cardinalities.  When several
hypergraphs are batched into one tensor they share the maximum over the
whole collection, which is the same rule applied to their disjoint union.

`rw_quantities_sparse` works from an incidence matrix B so it scales to
large datasets; the synthetic experiment builds B per small graph with
`build_incidence_scipy` and calls the same function, so there is exactly
one implementation of the formulas in the repository.
"""

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------
# Incidence matrix from a hyperedge list (TopoNetX-independent).
# # --------------------------------------------------------------------------
def build_incidence_scipy(n_nodes, hyperedges):
    """Build the N x M incidence matrix B from a list of hyperedges.

    Hyperedges of size < 2 are dropped (they do not contribute to the walk).
    """

    rows, cols = [], []
    col = 0
    for he in hyperedges:
        he = sorted(set(int(v) for v in he))
        if len(he) >= 2:
            for v in he:
                rows.append(v)
                cols.append(col)
            col += 1
    data = np.ones(len(rows), dtype=np.float64)
    return sp.coo_matrix((data, (rows, cols)),
                         shape=(n_nodes, col)).tocsr()


# --------------------------------------------------------------------------
# Clique expansion (used to verify the synthetic twin pairs).
# --------------------------------------------------------------------------
def clique_expansion(hyperedges, n_nodes=None):
    """Edge set (as a set of frozensets) of the pairwise graph induced by H."""

    edges = set()
    for e in hyperedges:
        e = tuple(sorted(set(int(v) for v in e)))
        if len(e) < 2:
            continue
        for i in range(len(e)):
            for j in range(i + 1, len(e)):
                edges.add(frozenset((e[i], e[j])))
    return edges


# --------------------------------------------------------------------------
# Random-walk quantities from the incidence matrix.
# --------------------------------------------------------------------------
def max_cardinality(B):
    """K := max_{e in E} |e|, over the hyperedges that carry transitions.

    Hyperedges of size < 2 are excluded, matching `rw_quantities_sparse`.
    Returns 2 for a hypergraph with no such hyperedge, so that the descriptor
    always has at least one bin.
    """

    card = np.asarray(B.tocsc().sum(axis=0)).ravel()
    card = card[card >= 2]
    return int(card.max()) if card.size else 2


def rw_quantities_sparse(B, max_card=None):
    """Compute edge_index, the three RW kernels and the two descriptors.

    Parameters
    ----------
    B : scipy sparse, shape [N, M]
        Incidence matrix, rows aligned to node ids 0..N-1.
    max_card : int or None
        The cardinality resolution K.  `None` (the default) selects the
        canonical choice K := max_{e in E} |e|, for which every observed
        cardinality 2, ..., K gets its own bin and the overflow bin never
        merges two distinct cardinalities.  An explicit smaller value folds
        all cardinalities >= max_card into the final bin and is only for
        deliberate truncation experiments; a larger value adds bins that no
        hyperedge can populate.  Batching several hypergraphs into one tensor
        requires a shared K, namely the maximum over the whole collection.

    Returns
    -------
    edge_index : int array [2, E]   row 0 = source u, row 1 = target v
    p          : dict "EN"/"EE"/"WE" -> float array [E], kernel at (v, u)
    chi        : dict "EE"/"WE"      -> float array [E, max_card-1]
    d_h        : float array [N]     hypergraph degree
    """

    B = B.tocsr().astype(np.float64)
    card = np.asarray(B.sum(axis=0)).ravel()       # cardinality per hyperedge
    keep = card >= 2
    B = B[:, keep]
    card = card[keep]

    if max_card is None:                           # K := max_e |e|
        max_card = int(card.max()) if card.size else 2

    w = 1.0 / (card - 1.0)                         # 1/(|e|-1)
    d_h = np.asarray(B.sum(axis=1)).ravel()        # d_H(v) = |E(v)|
    s_h = np.asarray(B @ (card - 1.0)).ravel()     # s_H(v) = sum_e (|e|-1)

    # Support of the adjacency: (B B^T)[v, u] = c_vu for v != u.  The indices
    # are sorted so that the edge list is in canonical (v, u) lexicographic
    # order: sparse products return unsorted indices whose order depends on
    # the operands, and float summation in the scatter is not associative, so
    # an uncanonical edge list makes results depend on an implementation
    # detail of the product.
    S = (B @ B.T).tocsr()
    S.sort_indices()
    S = S.tocoo()
    off = S.row != S.col                           # drop the diagonal
    src = S.col[off]                               # u
    tgt = S.row[off]                               # v

    # One sparse product per cardinality bin and per weighting.  Summing the
    # bins recovers Z^EE_vu and c_vu, so there is a single code path for the
    # descriptors and the pair masses that feed the kernels.
    n_bins = max_card - 1
    bin_idx = np.minimum(card.astype(int), max_card) - 2
    chi_ee = np.zeros((len(src), n_bins), dtype=np.float64)
    chi_we = np.zeros((len(src), n_bins), dtype=np.float64)
    for c in range(n_bins):
        sel = (bin_idx == c).astype(np.float64)
        if sel.sum() == 0:
            continue
        Mee = (B @ sp.diags(w * sel) @ B.T).tocsr()
        Mwe = (B @ sp.diags(sel) @ B.T).tocsr()
        chi_ee[:, c] = np.asarray(Mee[tgt, src]).ravel()
        chi_we[:, c] = np.asarray(Mwe[tgt, src]).ravel()

    z_vu = chi_ee.sum(axis=1)                      # Z^EE_vu
    c_vu = chi_we.sum(axis=1)                      # c_vu = |E(v,u)|
    chi_ee /= z_vu[:, None]
    chi_we /= c_vu[:, None]

    n_nb = np.bincount(tgt, minlength=B.shape[0]).astype(np.float64)  # |N_H(v)|

    p = {
        "EN": 1.0 / n_nb[tgt],
        "EE": z_vu / np.where(d_h > 0, d_h, 1.0)[tgt],
        "WE": c_vu / np.where(s_h > 0, s_h, 1.0)[tgt],
    }
    chi = {"EE": chi_ee, "WE": chi_we}

    edge_index = np.vstack([src, tgt]).astype(np.int64)
    return edge_index, p, chi, d_h


# Kernels that induce a cardinality descriptor (EN mediates no hyperedge).
KERNELS = ("EN", "EE", "WE")
CHI_KERNELS = ("EE", "WE")
