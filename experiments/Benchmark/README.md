# Experiment 3 - Realistic benchmark (co-citation node classification)

Node classification on the standard co-citation hypergraph benchmarks
(Cora, Citeseer). This experiment complements the synthetic and gadget
experiments: those isolate *what* the descriptors can do that clique
expansion provably cannot; this one measures whether that ability
translates into accuracy on real datasets with real features.

## What the experiment does

For every dataset, every model is trained from scratch and evaluated on a
held-out test set. Twelve models are compared:

| Group | Model | Structure used |
|---|---|---|
| control | `MLP` | none (features only) |
| graph | `Clique-GCN` | clique expansion of the hypergraph |
| graph | `Clique-GIN` | clique expansion of the hypergraph |
| hg-derived | `AllDeepSets (mean)` | incidence matrix; **mean** aggregation both ways (the published AllSet layer uses sums) |
| ablation | `A: EN` | uniform mean over neighbours, no descriptor |
| ablation | `B: EE` | `P^EE` kernel, no descriptor |
| ablation | `C: EN + χ^EE` | uniform mean + EE-induced descriptor |
| **proposed** | `D: EE-Pattern` | **`P^EE` + `χ^EE`** (the thesis EO-Pattern model) |
| control | `E: EE-Pattern, shuffled χ` | D with `χ^EE` globally permuted |
| ablation | `F: WE` | `P^WE` kernel, no descriptor |
| **proposed** | `G: WE-Pattern` | **`P^WE` + `χ^WE`** |
| control | `H: WE-Pattern, shuffled χ` | G with `χ^WE` globally permuted |

`A`-`H` is the full ablation ladder shared by every experiment in this
repository (`eompp.node_models.RW_SPECS`): it isolates the contribution of
the two hypergraph-derived ingredients for both the EE and WE paradigms.
`E` and `H` are negative controls — if shuffled `χ` did as well as real `χ`,
`χ` would be carrying no signal.

## Data source

The datasets are committed in `data/` for full reproducibility. See
`data/README.md` for provenance (HyperGCN commit hash + SHA-256 checksums).

| | nodes | features | classes | hyperedges | max `\|e\|` |
|---|---|---|---|---|---|
| Cora | 2,708 | 1,433 | 7 | 1,579 | 5 |
| Citeseer | 3,312 | 3,703 | 6 | 1,079 | 26 |

## Cardinality resolution `K`

`K := max_e |e|` is resolved automatically per dataset from the hypergraph
itself, inside `eompp.eo.rw_quantities_sparse` when `max_card=None`, not
fixed in advance: **K=5 for Cora,
K=26 for Citeseer**. This is the exact encoding described in the
accompanying note — no cardinality is folded into an overflow bin on either
dataset. The resolved value is recorded in `results_benchmark.json` under
`results["<dataset>"]["_meta"]["max_card"]`.

## Where SciPy and TopoNetX are used

The incidence matrix is built with the dependency-free SciPy builder in
`eompp.eo`, which is the one actually consumed downstream. When **TopoNetX**
(`ColoredHyperGraph`, from the TopoX suite) is installed, `data_benchmark.py`
additionally builds the incidence matrix that way and cross-checks that both
builders represent the same hyperedges (as column sets) — this validates our
row-reordering glue, not TopoNetX itself. SciPy is used regardless of
whether TopoNetX is installed, so results do not depend on the environment.

## Files

| File | Purpose |
|---|---|
| `data/` | committed datasets + provenance (`data/README.md`) |
| `data_benchmark.py` | data loading, incidence construction (SciPy + TopoNetX cross-check), split |
| `experiment_benchmark.py` | config + multi-seed driver (resumable) |
| `results_benchmark.json` | per-dataset results, all seeds, resolved K in `_meta` |
| `requirements_benchmark.txt` | dependencies (incl. TopoNetX) |
| `thesis/` | numbers as submitted in the bachelor's thesis, kept for reference |

The twelve models and the train/eval routines live in the shared `eompp`
package (`eompp.node_models`, `eompp.node_training`) — see the root README.

## How to run

```bash
pip install -e ..                              # installs the eompp package
pip install -r requirements_benchmark.txt      # adds TopoNetX (optional, cross-check only)

# full benchmark, both datasets
python experiment_benchmark.py --cpu

# a single dataset
python experiment_benchmark.py --cpu --dataset cora

# fast smoke test
python experiment_benchmark.py --cpu --quick
```

The driver is resumable: rerunning the same command skips every
`(model, seed)` pair already present in `results_benchmark.json`. `--out`
writes to a different file instead of merging into the canonical one; use
it to reproduce from scratch, or to split Cora and Citeseer into two
parallel processes (see the root README for the parallel-run pattern and
merge step).

## Results

Test accuracy (%), mean ± std over 20 seeds. *all*: accuracy over all
nodes. *active*: accuracy restricted to nodes with `d_H(v) > 0` (53% of
Cora, 44% of Citeseer nodes are active). Hidden width 128, 2 layers,
lr = 1e-3.

### Cora (K = 5)

| Model | all | active |
|---|---|---|
| MLP | 73.8 ± 1.4 | 75.1 ± 2.0 |
| **Clique-GCN** | **79.9 ± 1.5** | **85.0 ± 1.7** |
| Clique-GIN | 76.6 ± 1.3 | 82.8 ± 1.5 |
| AllDeepSets (mean) | 75.2 ± 1.5 | 76.5 ± 1.9 |
| A: EN | 76.3 ± 1.2 | 80.4 ± 1.7 |
| B: EE | 76.5 ± 1.2 | 80.8 ± 1.5 |
| C: EN + χ^EE | 76.5 ± 1.2 | 80.7 ± 1.7 |
| D: EE-Pattern | 76.2 ± 1.2 | 80.5 ± 1.8 |
| E: EE-Pattern, shuffled χ | 76.3 ± 1.3 | 80.5 ± 1.8 |
| F: WE | 76.4 ± 1.0 | 80.7 ± 1.5 |
| G: WE-Pattern | 76.2 ± 1.0 | 80.4 ± 1.6 |
| H: WE-Pattern, shuffled χ | 76.1 ± 1.3 | 80.2 ± 1.6 |

### Citeseer (K = 26)

| Model | all | active |
|---|---|---|
| MLP | 72.2 ± 0.9 | 74.9 ± 1.9 |
| **Clique-GCN** | **73.2 ± 1.4** | **76.3 ± 2.1** |
| Clique-GIN | 71.2 ± 1.0 | 73.9 ± 2.2 |
| AllDeepSets (mean) | 73.1 ± 1.0 | 76.0 ± 1.8 |
| A: EN | 72.4 ± 1.0 | 75.7 ± 1.7 |
| B: EE | 72.5 ± 1.0 | 75.9 ± 1.5 |
| C: EN + χ^EE | 72.5 ± 1.1 | 75.7 ± 2.1 |
| D: EE-Pattern | 72.7 ± 1.0 | 75.8 ± 2.1 |
| E: EE-Pattern, shuffled χ | 72.6 ± 1.0 | 75.7 ± 2.2 |
| F: WE | 72.5 ± 1.0 | 75.9 ± 1.6 |
| G: WE-Pattern | 72.5 ± 1.0 | 75.7 ± 2.0 |
| H: WE-Pattern, shuffled χ | 72.6 ± 1.0 | 75.8 ± 1.9 |

### Reading the results

This is a **negative result for both descriptors on co-citation data**, and
it holds on both datasets, not just the one the thesis originally tested.
On Cora, the eight-variant ladder A–H spans 0.46 points (76.06–76.52); on
Citeseer, 0.29 points (72.42–72.70) — Citeseer is *flatter*, despite having a
much wider range of hyperedge cardinalities (K=26 vs K=5), which rules out
"the descriptor didn't have enough resolution" as an explanation. `E ≈ D`
and `H ≈ G` on both datasets: shuffling the descriptor changes nothing,
meaning the model is not using its structured organization.

The structural reason: about four fifths of adjacent ordered co-citation
pairs share exactly one hyperedge (`c_vu = 1`) on both datasets (80.9% on
Cora, 80.0% on Citeseer). On such a pair `χ^EE_vu = χ^WE_vu` identically,
so the two descriptors agree on 85.9% (Cora) and 83.5% (Citeseer) of all
adjacent pairs. The WE kernel degenerates onto EN at the level of vertices
rather than pairs: `P^WE(v,·) = P^EN(v,·)` exactly when `c_vu` is *constant*
over `u` in `N_H(v)`, since then `s_H(v) = c·|N_H(v)|` and the common factor
`c` cancels — this holds for 59.8% of active Cora vertices and 71.0% of
Citeseer ones (the stricter condition `c_vu = 1` throughout accounts for
58.4% and 67.4%). Either way
the three walks collapse onto (almost) the same object before the model
ever sees the data. Clique-GCN's residual advantage on Cora (not on
Citeseer) is not explained by this and is left as such.

47% of Cora nodes and 56% of Citeseer nodes satisfy `d_H(v)=0` and receive
no messages from any structural model; the *all* column mixes these in with
active nodes.

## Note on compute

Both datasets run on CPU. The two datasets are independent and can be run
in parallel (`--dataset cora --out results_cora_only.json` /
`--dataset citeseer --out results_citeseer_only.json`, run concurrently,
then merged) — see the root README.
