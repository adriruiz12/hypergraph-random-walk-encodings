# Experiments 4 and 5 - Complementarity

This folder holds two mirror-image gadgets, each isolating the one regime
the other experiments never reach: a task where **both** hypergraph-derived
ingredients of a Pattern layer are simultaneously load-bearing.

* `pqr.py`    - built against the **EE** factorization `(P^EE, χ^EE)`.
* `pqr_we.py` - built against the **WE** factorization `(P^WE, χ^WE)`.

Running both is the point: a gadget built against one factorization is not
neutral with respect to the other paradigm's kernel, so neither experiment
alone can support a general complementarity claim. Read together, they show
that complementarity is a property of the *chosen factorization*, not of
the hypergraph information itself (Proposition "Pairwise information
equivalence" in the accompanying note).

`pqr_we.py --m-decoys 2` is a dilution control: the WE gadget attaches more
filler mass per target than the EE gadget by default (`m=4`), so the
informative neighbour carries a smaller share of the aggregation weight.
The control repeats the experiment at matched dilution (`m=2`), fixed by an
analytical computation, not chosen from the results.

## Experiment 4: the EE gadget (`pqr.py`)

For every target node `v`: one informative neighbour `Q`, connected through
two size-3 hyperedges; `m` decoys of type `P`, each through one size-2
hyperedge; `m` decoys of type `R`, each through one size-3 hyperedge;
neutral fillers complete the hyperedges.

| neighbour | construction | `Z^EE_vu` | `c_vu` |
|---|---|---|---|
| **P** | one size-2 hyperedge | 1 | 1 |
| **Q** | two size-3 hyperedges | 1 | 2 |
| **R** | one size-3 hyperedge | 1/2 | 1 |

`χ^EE` separates P from `{Q, R}` (different cardinality) but not Q from R
(same cardinality, so same bin). `P^EE` (via `Z^EE`) separates R from
`{P, Q}` (different mass) but not P from Q. Only the pair `(Z^EE, χ^EE)`
jointly singles out Q. `Q`'s feature carries the true label of `v`; `P` and
`R` are decoys with random labels.

`K := max_e |e| = 3` (only cardinalities 2 and 3 occur), resolved
automatically.

### Results

Mean ± std over ten seeds; 450 gadgets, `m=4`, width 64, one layer.

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

**Reading.** Neither ingredient is individually strong: `P^EE` alone gives
+5.5, and `χ^EE` alone, attached to the EN kernel, is not merely unhelpful
but slightly harmful (−1.6). Their combination gives +17.4, and shuffling
`χ^EE` collapses D back to roughly the kernel-only level. `D` and `G` land
almost exactly together (80.1 vs 80.0) — a direct empirical illustration of
the pair-level information equivalence between the two Pattern layers.

The gadget is *not* neutral with respect to WE: Q is the only neighbour
with `c_vu = 2`, and `P^WE = c_vu / s_H(v)` is exactly the kernel that
retains multiplicity. The pure WE layer (F) reaches +13.1 with no
descriptor at all — the discrimination the EE paradigm can only perform
jointly, the WE kernel performs on its own.

## Experiment 5: the WE gadget (`pqr_we.py`)

The mirror construction. `Q`: two size-3 hyperedges; `P`-decoys: two size-4
hyperedges each; `R`-decoys: one size-3 hyperedge each.

| neighbour | construction | `c_vu` | `Z^EE_vu` |
|---|---|---|---|
| **P** | two size-4 hyperedges | 2 | 2/3 |
| **Q** | two size-3 hyperedges | 2 | 1 |
| **R** | one size-3 hyperedge | 1 | 1/2 |

`P^WE` (via `c_vu`) collapses Q with P; `χ^WE` collapses Q with R; only the
pair singles out Q. Symmetrically, `Z^EE` is distinct for all three types,
so `P^EE` alone identifies Q — the mirror of experiment 4.

`K := max_e |e| = 4` here, resolved automatically.

At `m=4`, Q carries only `P^WE(v,u_Q) = 2/36 ≈ 5.6%` of the aggregation
mass, against `1/10 = 10%` in the EE gadget. The `m=2` control matches this
share exactly (`2/20 = 10%`).

### Results, `m=4` (as specified)

| Model | Accuracy | Δ vs A |
|---|---|---|
| MLP | 54.4 ± 1.3 | |
| Clique-GCN | 65.2 ± 2.7 | |
| Clique-GIN | 62.7 ± 4.3 | |
| AllDeepSets (mean) | 54.9 ± 0.0 | |
| A: EN | 55.9 ± 3.2 | — |
| B: EE | 72.7 ± 7.1 | +16.7 |
| C: EN + χ^EE | 54.9 ± 0.0 | −1.1 |
| D: EE-Pattern | 73.6 ± 10.4 | +17.7 |
| E: EE-Pattern, shuffled χ | 68.2 ± 9.4 | +12.3 |
| F: WE | 63.8 ± 6.5 | +7.9 |
| G: WE-Pattern | 65.3 ± 12.9 | +9.4 |
| H: WE-Pattern, shuffled χ | 60.5 ± 7.0 | +4.6 |

At this dilution, `C` collapses to the majority-class predictor
(54.9 ± 0.0), essentially indistinguishable from `A` — `χ^EE` confuses Q
with R here just as EN does. The intended super-additivity on the WE side
is present in direction (`G > F > H`), but the `G − F` gap (+1.5) sits well
inside a standard deviation of 12.9 and is not established at this
dilution.

### Results, `m=2` (dilution-matched control)

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

**Reading.** At matched dilution the mirror closes cleanly. `P^WE` alone
adds +4.6; its shuffled control falls *further*, to +2.1, rather than
merely matching F (suggesting the shuffled descriptor mildly interferes
rather than being wholly neutral); the combination reaches +20.6 — the
same shape of super-additivity as the +17.4 of experiment 4, and by a
paired test across seeds, G is significantly above both F and H (p<0.005).
This is the single clearest piece of evidence in either gadget that the WE
factorization is not redundant with the EE one. Symmetrically, `E ≈ D`
here (86.5 vs 87.7): given `P^EE`, `χ^EE` is close to inert, because
`Z^EE` already identifies Q on its own.

## Models

The same twelve models as the benchmark (from `eompp.node_models`,
unchanged): MLP, Clique-GCN, Clique-GIN, AllDeepSets, and the eight-variant
A–H ladder.

## Files

| File | Purpose |
|---|---|
| `pqr.py` | EE gadget generator + target-only split + driver |
| `pqr_we.py` | WE gadget generator + target-only split + driver |
| `results_complementarity.json` | EE gadget results (per-seed scores included) |
| `results_complementarity_we.json` | WE gadget results, `m=4` (per-seed scores included) |
| `results_complementarity_we_m2.json` | WE gadget results, `m=2` dilution control |

The random-walk quantities, the twelve models, and the train/eval routines
all come from the shared `eompp` package — see the root README.

## How to run

```bash
pip install -e ..                              # installs the eompp package

# EE gadget (reproduces the table above)
python pqr.py --n-gadgets 450 --m-decoys 4 --n-seeds 10 --n-layers 1

# WE gadget, both dilutions
python pqr_we.py --n-gadgets 450 --m-decoys 4 --n-seeds 10
python pqr_we.py --n-gadgets 450 --m-decoys 2 --n-seeds 10 \
    --out results_complementarity_we_m2.json

# inspect the gadget signatures without training
python -c "from pqr import make_pqr_dataset, check_signatures; \
check_signatures(make_pqr_dataset(n_gadgets=5))"
```

The hypergraph, labels, and node features are generated once using
`seed=12345` and remain fixed across runs. Each run seed controls the
train/val/test split, model initialization, and dropout masks; for the
shuffled-control variants, it also controls the global permutation of `χ`.
