# Experiment 1 - Cardinality separation (twin hypergraphs)

Graph-classification experiment for the full EN/EE/WE family.

It answers one question, in a fully controlled setting:

> **Do the EE- and WE-conditioned cardinality descriptors detect hyperedge
> structure that a clique expansion provably destroys?**

## The idea

Every example is a *graph-classification* instance. For each random
**base graph** `G` (built as a union of cliques) we produce **two
hypergraphs on the same vertex set**:

| class | hypergraph | hyperedges |
|---|---|---|
| 0 - "big" | `H_big` | the original cliques, one hyperedge each |
| 1 - "small" | `H_small` | each clique replaced by all its pairwise edges |

**By construction `clique_expansion(H_big) == clique_expansion(H_small)`**
(asserted for every twin pair). Node features are shared between twins and
constant. Hence any model that is a function of the clique expansion alone,
or of a row-stochastic aggregation of constant features, receives
*identical* input for the two classes and, since the dataset contains both
twins, **cannot exceed 50% accuracy**. The two hypergraphs differ only in
how the same pairwise support is grouped into hyperedges of different
cardinalities — the distinction the cardinality descriptors are built to
expose. Splitting is by base graph, so the 50%-bound holds inside every
split.

## The models (the ablation ladder)

Ten models: two clique-expansion baselines plus the eight-variant ladder
shared by every experiment in this repository (`eompp.node_models.RW_SPECS`,
graph-classification wrappers around the shared `eompp` layers).

| name | weights | message input |
|---|---|---|
| Clique-GCN | GCN norm | `h_u` |
| Clique-GIN | GIN sum | `h_u` |
| A: EN | uniform mean | `h_v, h_u` |
| B: EE | `P^EE` | `h_v, h_u` |
| C: EN + χ^EE | uniform mean | `h_v, h_u, χ^EE` |
| D: EE-Pattern | `P^EE` | `h_v, h_u, χ^EE` |
| E: EE-Pattern, shuffled χ | `P^EE` | `h_v, h_u, χ^EE` (permuted) |
| F: WE | `P^WE` | `h_v, h_u` |
| G: WE-Pattern | `P^WE` | `h_v, h_u, χ^WE` |
| H: WE-Pattern, shuffled χ | `P^WE` | `h_v, h_u, χ^WE` (permuted) |

## Results

Test accuracy, mean ± std over **ten** seeds (`feature_mode=constant`):

| Model | Accuracy (%) | Macro-F1 (%) |
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

Reading of the table:

* **Clique-GCN / GIN / A / B / F at 50%**: any model that only sees the
  clique expansion is at chance by construction. The three kernels *do*
  differ between twins (`P^EE` and `P^WE` are not equal here), but with
  constant node features a row-stochastic weight maps a constant input to
  that same constant regardless of its values — only a quantity entering
  `ψ` as a feature can break the symmetry.
* **C, D, G at 100%**: what is provable is *separability*, not the accuracy.
  For a pair contained in a single clique of size `s`,
  `χ^EE_vu = χ^WE_vu = c_s` in `H_big` and `c_2` in `H_small`, so the two
  classes receive disjoint descriptor values and the task is linearly
  separable on the mean-pooled readout. That the optimizer actually attains
  100% on all ten seeds is the empirical outcome of the committed run.
* **E, H at 50%**: permuting `χ` across pairs destroys the signal, so the
  gain is genuine hypergraph structure, not extra input capacity.
* **Clique-GIN's macro-F1 (38.1 ± 7.0)** is the one chance-level row not
  pinned at 33.3: its 50% accuracy is not always reached by collapsing onto
  a single class. The accuracy bound is unaffected.

## Files

| file | content |
|---|---|
| `data_synthetic.py` | twin-pair dataset generator and base-graph split |
| `models_synthetic.py` | graph-classification wrappers (pooling + MLP head) over `eompp` layers |
| `experiment_synthetic.py` | disjoint-union batching, training, multi-seed evaluation, results table |
| `results_synthetic.json` | results (full config + per-seed accuracies) |

The random-walk quantities (`P^EN`, `P^EE`, `P^WE`, `χ^EE`, `χ^WE`) and the
message-passing layers come from the shared `eompp` package — see the root
README.

## Running

```bash
pip install -e ..                              # installs the eompp package

python experiment_synthetic.py                 # full run (constant features)
python experiment_synthetic.py --quick         # fast sanity check
python experiment_synthetic.py --feature-mode random
```

`K := max_e |e|` is resolved automatically from the whole dataset (all
splits are batched into one tensor and must share one `χ` dimension); the
resolved value is written back into the results file's `config.max_card`.

## Design notes

* **Selection criterion.** `train_one` selects the best checkpoint on
  `(val accuracy, -val loss)`, not accuracy alone. On this task validation
  accuracy saturates at 1.0 almost immediately (a linearly-separable
  problem with margin 1), after which it can never improve strictly again;
  without the loss tie-break the first, still under-converged model to
  reach that ceiling would be kept for the rest of training. This affects
  only how the checkpoint is selected, not what the task is.
* **Pure PyTorch.** No PyG / DGL. The synthetic graphs are small and a
  self-contained implementation is easier to audit.
* **Why B at chance is the *right* result.** With constant features it is
  provable that any row-stochastic weight is inert, so the descriptor
  signal can only enter through `χ` (the sole ingredient `ψ` consumes as a
  feature). This is the clean structural reason, not a numerical
  cancellation of `P^EE` (or `P^WE`) between the twins, which does not hold
  in general — the twins' kernels genuinely differ.
