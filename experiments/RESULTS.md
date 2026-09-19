# Results

All numbers are test accuracy in percent, mean ± std over seeds, produced by
the scripts in this repository.  Regenerate this listing with
`python summarise.py`.

The eight lettered variants are defined once in `eompp.node_models.RW_SPECS`:

| Variant | Kernel | Descriptor |
|---|---|---|
| A | `P^EN` | none |
| B | `P^EE` | none |
| C | `P^EN` | `χ^EE` |
| D | `P^EE` | `χ^EE`  (EE-Pattern; the EO-Pattern model of the thesis) |
| E | `P^EE` | shuffled `χ^EE` |
| F | `P^WE` | none |
| G | `P^WE` | `χ^WE`  (WE-Pattern) |
| H | `P^WE` | shuffled `χ^WE` |

---

## Two structural identities used throughout

**(i) On an `r`-uniform hypergraph the two hyperedge-mediated paradigms
coincide.**  Since `Z^EE_vu = c_vu/(r-1)` and `s_H(v) = (r-1) d_H(v)`,

```
P^EE(v,u) = c_vu / ((r-1) d_H(v)) = P^WE(v,u),     χ^EE_vu = χ^WE_vu = c_r.
```

**(ii) At the pair level the two Pattern layers carry the same information.**
Writing `n_r(v,u)` for the number of shared hyperedges of size `r`, and
provided the cardinality resolution satisfies `K ≥ max_e |e|`,

```
(c_vu, χ^WE_vu)  <->  (n_r)_r  <->  (Z^EE_vu, χ^EE_vu)
```

is a bijection.  The condition on `K` is not cosmetic: with `K` truncated
below the true maximum the overflow bin merges distinct cardinalities, two
hyperedges of different size can map to the same `(n_r)_r`, and the
reconstruction of `(n_r)_r` from `(Z^EE, χ^EE)` no longer agrees with the
direct count.  Every experiment here resolves `K` from the data, so the
condition holds throughout.

Given the bijection, EE-Pattern and WE-Pattern differ only in the vertex
normalisation (`d_H` versus `s_H`) and in how the same pair information is
split between an aggregation weight and a message feature.  This is why the
experiments below come in mirror pairs.

---

## 1. Cardinality twins (`Synthetic/`, graph classification, 10 seeds)

Twin hypergraphs with identical clique expansion and identical node features,
differing only in hyperedge cardinality.  Node features are constant, so by
feature degeneracy every row-stochastic kernel is inert.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Clique-GCN | 50.0 ± 0.0 | 33.3 ± 0.0 |
| Clique-GIN | 50.0 ± 0.0 | 38.1 ± 7.0 |
| A: EN | 50.0 ± 0.0 | 33.3 ± 0.0 |
| B: EE | 50.0 ± 0.0 | 33.3 ± 0.0 |
| **C: EN + χ^EE** | **100.0 ± 0.0** | **100.0 ± 0.0** |
| **D: EE-Pattern** | **100.0 ± 0.0** | **100.0 ± 0.0** |
| E: EE-Pattern, shuffled χ | 50.0 ± 0.0 | 33.3 ± 0.0 |
| F: WE | 50.0 ± 0.0 | 33.3 ± 0.0 |
| **G: WE-Pattern** | **100.0 ± 0.0** | **100.0 ± 0.0** |
| H: WE-Pattern, shuffled χ | 50.0 ± 0.0 | 33.3 ± 0.0 |

**Reading.** Both descriptors recover cardinality structure that the clique
expansion provably destroys; no kernel can, and the shuffled controls return
to chance.  The three kernels sit at chance for the same reason and not
because they are equal: they do differ between the twins, but a row-stochastic
weight maps constant features to the same constant.

Clique-GIN's macro-F1 (38.1 ± 7.0) is the one chance-level row not pinned at
33.3: its 50 % accuracy is not always reached by collapsing onto a single
class, as it is in every other row at chance.  The accuracy bound is
unaffected.

## 2. Multiplicity (`Multiplicity/`, node classification, 10 seeds)

3-uniform gadget: the informative neighbour is the only one sharing two
hyperedges with the target.  By identity (i), both descriptors are provably
constant and the two kernels provably coincide.

| Model | Accuracy | Δ vs A |
|---|---|---|
| MLP | 54.0 ± 0.0 | |
| Clique-GCN | 70.0 ± 3.8 | |
| Clique-GIN | 70.4 ± 3.5 | |
| AllDeepSets (mean) | 54.1 ± 0.3 | |
| A: EN | 70.4 ± 2.7 | — |
| **B: EE** | **86.8 ± 2.2** | +16.4 |
| C: EN + χ^EE | 69.6 ± 4.4 | −0.9 |
| D: EE-Pattern | 85.2 ± 4.2 | +14.8 |
| E: EE-Pattern, shuffled χ | 85.2 ± 4.2 | +14.8 |
| **F: WE** | **86.8 ± 2.2** | +16.4 |
| G: WE-Pattern | 85.2 ± 4.2 | +14.8 |
| H: WE-Pattern, shuffled χ | 85.2 ± 4.2 | +14.8 |

**Reading.** The exact mirror of experiment 1: the kernels separate and no
descriptor can.  `B ≡ F` and `D ≡ E ≡ G ≡ H` hold exactly — seed by seed, not
merely in the mean — as identity (i) requires; under 3-uniformity those
variants are literally the same model, so the agreement is a correctness check
rather than a finding.  `C ≈ A` is the structural-inertness proposition
observed in practice.

## 3. Co-citation benchmark (`Benchmark/`, node classification, 20 seeds)

Cora and Citeseer, random per-class 50/25/25 splits, hidden 128, two layers.
All twelve models have 20/20 seeds on both datasets; `Benchmark/README.md`
holds the two accuracy tables and `results_benchmark.json` the per-seed
scores.  `thesis/` records the numbers as submitted in the bachelor's thesis.

Structural diagnostics measured on the data (see the note below):

| | Cora | Citeseer |
|---|---|---|
| adjacent ordered pairs with `c_vu = 1` | 80.9 % | 80.0 % |
| pairs where `χ^EE = χ^WE` | 85.9 % | 83.5 % |
| pairs where `P^WE = P^EN` | 33.6 % | 37.6 % |
| pairs where `P^EE = P^EN` | 21.3 % | 29.6 % |
| active vertices with `c_vu` constant over `N_H(v)` (⇔ `P^WE = P^EN`) | 59.8 % | 71.0 % |
| active vertices with `c_vu = 1` throughout `N_H(v)` | 58.4 % | 67.4 % |
| mean \|`P^WE` − `P^EN`\| | 0.024 | 0.018 |
| mean \|`P^EE` − `P^EN`\| | 0.042 | 0.036 |
| `K` = max_e \|`e`\| | 5 | 26 |

**Reading.** Four fifths of adjacent ordered pairs share exactly one
hyperedge, and on every such pair `χ^EE_vu = χ^WE_vu` identically — this is
what drives the two descriptors to agree on 85.9 % (Cora) and 83.5 %
(Citeseer) of all adjacent pairs.

The kernels degenerate on a different scale.  `P^WE(v,u) = P^EN(v,u)` does
*not* follow from `c_vu = 1` at that pair: only 40.5 % (Cora) and 45.2 %
(Citeseer) of the `c_vu = 1` pairs satisfy it.  The collapse of the WE kernel
onto EN is a **per-vertex** statement: `P^WE(v,·) = P^EN(v,·)` exactly when
`c_vu` is *constant* over `u` in `N_H(v)` — then `s_H(v) = c·|N_H(v)|` and the
common factor `c` cancels.  That holds for 59.8 % of active Cora vertices and
71.0 % of active Citeseer ones (the stricter condition `c_vu = 1` throughout
accounts for 58.4 % and 67.4 %). So the descriptors coincide pair by pair, the
kernels vertex by vertex, and either way the three walks collapse onto
(almost) the same object before the model ever sees the data.

The submitted thesis already reported a flat A–E ladder here; these
diagnostics predict, before training, that the F–H rows should be flat as
well, so a flat WE ladder confirms that the co-citation regime is degenerate
for all three walks rather than unfavourable to one of them.

## 4. Complementarity, EE factorization (`Complementarity/pqr.py`, 10 seeds)

Signatures per neighbour type: `P = (|e|=2, Z=1, c=1)`, `Q = (|e|=3, Z=1,
c=2)`, `R = (|e|=3, Z=1/2, c=1)`.  `χ^EE` collapses Q with R, `P^EE` collapses
P with Q, only the pair singles out Q.

| Model | Accuracy | Δ vs A |
|---|---|---|
| MLP | 54.2 ± 2.1 | |
| Clique-GCN | 63.5 ± 3.8 | |
| Clique-GIN | 63.3 ± 3.5 | |
| AllDeepSets (mean) | 54.9 ± 0.0 | |
| A: EN | 62.7 ± 4.0 | — |
| B: EE | 68.1 ± 2.9 | +5.5 |
| C: EN + χ^EE | 61.1 ± 3.6 | −1.6 |
| **D: EE-Pattern** | **80.1 ± 3.1** | **+17.4** |
| E: EE-Pattern, shuffled χ | 68.1 ± 3.2 | +5.4 |
| F: WE | 75.8 ± 3.1 | +13.1 |
| G: WE-Pattern | 80.0 ± 2.5 | +17.3 |
| H: WE-Pattern, shuffled χ | 75.7 ± 2.8 | +13.0 |

**Reading.** Super-additivity for the EE factorization: +5.5 from the kernel
alone and −1.6 from the descriptor alone, +17.4 jointly, and the shuffled
control returns to the kernel-only level.  But the gadget is not neutral with
respect to WE: Q is the only neighbour with `c_vu = 2`, and `P^WE` is exactly
the kernel that retains multiplicity, so the pure WE layer reaches +13.1 with
no descriptor at all.  `D` and `G` land together (80.1 vs 80.0), an empirical
illustration of the pair-level information equivalence (ii).

## 5. Complementarity, WE factorization (`Complementarity/pqr_we.py`, 10 seeds)

The mirror construction: `P = (|e|=4, c=2)`, `Q = (|e|=3, c=2)`, `R = (|e|=3,
c=1)`.  `P^WE` collapses P with Q, `χ^WE` collapses Q with R.  The EE masses
are `Z_P = 2/3`, `Z_Q = 1`, `Z_R = 1/2`, all distinct, so `P^EE` alone
identifies Q.

### 5a. As specified, `m = 4` decoys per type

| Model | Accuracy | Δ vs A |
|---|---|---|
| MLP | 54.4 ± 1.3 | |
| Clique-GCN | 65.2 ± 2.7 | |
| Clique-GIN | 62.7 ± 4.3 | |
| AllDeepSets (mean) | 54.9 ± 0.0 | |
| A: EN | 55.9 ± 3.2 | — |
| B: EE | 72.7 ± 7.1 | +16.7 |
| C: EN + χ^EE | 54.9 ± 0.0 | −1.1 |
| **D: EE-Pattern** | **73.6 ± 10.4** | **+17.7** |
| E: EE-Pattern, shuffled χ | 68.2 ± 9.4 | +12.3 |
| F: WE | 63.8 ± 6.5 | +7.9 |
| G: WE-Pattern | 65.3 ± 12.9 | +9.4 |
| H: WE-Pattern, shuffled χ | 60.5 ± 7.0 | +4.6 |

The mirror prediction on the EE side is confirmed: `P^EE` alone gives the
largest single-ingredient effect (+16.7), exactly as `P^WE` did in experiment
4.  `C` collapses to the majority-class predictor, indistinguishable from `A`,
because `χ^EE` confuses Q with R here just as EN does.  The intended WE
super-additivity is present in direction (`G > F > H`), but the `G − F` gap of
+1.5 sits well inside a standard deviation of 12.9 (paired *t* across seeds,
p = 0.78) and is not established at this dilution.

### 5b. Dilution control, `m = 2` decoys per type

The WE gadget attaches two size-4 hyperedges per P-decoy, so at `m = 4` the
informative neighbour carries `P^WE(v,uQ) = 2/36 = 5.6 %` of the aggregation
mass, against `1/10 = 10 %` in experiment 4.  At `m = 2` it carries
`2/20 = 10 %`, matching experiment 4.  The setting is fixed by that
computation, not chosen from the results.

| Model | Accuracy | Δ vs A |
|---|---|---|
| MLP | 54.0 ± 0.0 | |
| Clique-GCN | 75.3 ± 4.7 | |
| Clique-GIN | 69.6 ± 3.8 | |
| AllDeepSets (mean) | 54.0 ± 0.0 | |
| A: EN | 70.3 ± 3.4 | — |
| B: EE | 85.6 ± 4.1 | +15.3 |
| C: EN + χ^EE | 73.7 ± 7.1 | +3.5 |
| D: EE-Pattern | 87.7 ± 2.3 | +17.4 |
| E: EE-Pattern, shuffled χ | 86.5 ± 2.0 | +16.3 |
| F: WE | 74.9 ± 2.8 | +4.6 |
| **G: WE-Pattern** | **90.9 ± 1.5** | **+20.6** |
| H: WE-Pattern, shuffled χ | 72.4 ± 6.8 | +2.1 |

**Reading.** At matched dilution the mirror closes.  `P^WE` alone gives +4.6,
the shuffled control neutralises the descriptor and falls further, to +2.1,
and the combination gives +20.6 — the same shape of super-additivity as the
+17.4 of experiment 4.  Across seeds the gap is significant in both
directions: G vs F, paired *t* p = 1.1e-8, Wilcoxon p = 0.002; G vs H, paired
*t* p = 1.4e-5, Wilcoxon p = 0.002 (0.002 is the smallest value the
signed-rank test can return at n = 10).  Symmetrically, `E ≈ D` here: given
`P^EE`, the descriptor `χ^EE` is close to inert, because `Z^EE` already
identifies Q on its own.

---

## What the five experiments support jointly

1. Cardinality and multiplicity are independent axes, and experiments 1 and 2
   isolate one each with the other neutralised by construction rather than by
   tuning.
2. By identity (ii), the two Pattern layers receive the same pair information;
   they differ in how it is split between weight and feature.
3. Consequently complementarity is a property of the chosen factorization, not
   of the hypergraph information, and experiments 4 and 5 demonstrate this in
   both directions: each gadget needs both ingredients of the paradigm it was
   built against, while the other paradigm resolves it with its kernel alone.

The three random walks therefore do not carry different information about a
pair of vertices.  They carry different **factorizations** of the same
information, and which factorization suits a task is a property of the task.
