# Random-walk-induced message passing on hypergraphs - experiments

Code for the experiments on the three vertex-level random-walk message
passing paradigms on hypergraphs:

| Paradigm | Kernel | Descriptor |
|---|---|---|
| **EN** (equal-nodes)     | `P^EN(v,u) = 1/|N_H(v)|`      | none (EN mediates no hyperedge) |
| **EE** (equal-edges)     | `P^EE(v,u) = Z^EE_vu/d_H(v)`  | `χ^EE` (1/(\|e\|-1)-weighted) |
| **WE** (weighted-edges)  | `P^WE(v,u) = c_vu/s_H(v)`     | `χ^WE` (uniformly weighted) |

The **EE-Pattern** layer (`P^EE` + `χ^EE`) is the EO-Pattern model of the
submitted bachelor's thesis; **EN** is the uniform aggregation baseline used
there; **WE** and **WE-Pattern** are new.

Five experiments.  They are designed in mirror pairs rather than as three
copies of the same protocol: each isolates one axis of hypergraph structure,
and each construction targeted at one paradigm has a counterpart targeted at
the other.

|            Script             |            Experiment            |                                 Question                                   |         Task         |
|-------------------------------|----------------------------------|----------------------------------------------------------------------------|----------------------|
| `Synthetic/experiment_synthetic.py` | 1 - Cardinality separation | Do the descriptors detect cardinality structure a clique expansion destroys? | graph classification |
| `Multiplicity/multiplicity.py`      | 2 - Multiplicity separation | Do the kernels detect multiplicity at provably constant `χ`? | node classification |
| `Benchmark/experiment_benchmark.py` | 3 - Realistic benchmark | Does any of this translate into accuracy on real co-citation data? | node classification |
| `Complementarity/pqr.py`            | 4 - Complementarity, EE | Are `P^EE` and `χ^EE` jointly load-bearing? | node classification |
| `Complementarity/pqr_we.py`         | 5 - Complementarity, WE | Are `P^WE` and `χ^WE` jointly load-bearing? | node classification |

Experiments 1 and 2 are mirror images: 1 varies cardinality at fixed
adjacency and multiplicity, 2 varies multiplicity at fixed adjacency and
cardinality.  Experiments 4 and 5 are mirror images: each gadget is built
against one factorization, and the other paradigm resolves it with its kernel
alone.  Read together they show that complementarity is a property of the
chosen factorization, not of the hypergraph information.

## Shared core: the `eompp` package

The method lives in the installable `eompp/` package; the three experiments
share a single implementation of the random-walk formulas and the
message-passing layers:

|         Module          |                                            Contents                                             |
|-------------------------|-------------------------------------------------------------------------------------------------|
|      `eompp/eo.py`      | `rw_quantities_sparse` (the one implementation of the three kernels and two descriptors), `clique_expansion`, `build_incidence_scipy` |
|    `eompp/layers.py`    |     `scatter_sum/mean`, `mlp`, `GCNLayer`, `GINLayer`, `AllDeepSetsLayer`, `RWPatternLayer`     |
|   `eompp/metrics.py`    |                                           `macro_f1`                                            |
| `eompp/node_models.py`  |         the twelve node-classification models + `RW_SPECS` + `build_model` + `MODEL_SPECS`      |
| `eompp/node_training.py`|                      `to_tensors`, `train_one`, `evaluate`, `shuffled_chi`                      |
|   `eompp/gadgets.py`    |             `split_targets`, `run_gadget` (shared driver for experiments 2, 4, 5)               |

The training harness is task-specific and not shared: graph classification
(Synthetic) uses disjoint-union batching with per-graph pooling; node
classification (Benchmark, Multiplicity, Complementarity) uses a masked
single graph.  The node-classification experiments share `eompp.node_*`, the
three controlled gadgets additionally share `eompp.gadgets`, and the
graph-classification harness lives in `Synthetic/`.

## Ablation ladder

Every experiment runs the same eight lettered variants, defined once in
`eompp.node_models.RW_SPECS`:

| Variant | Kernel | Descriptor | Role |
|---|---|---|---|
| A | `P^EN` | none | pure EN layer (the uniform baseline of the thesis) |
| B | `P^EE` | none | pure EE layer |
| C | `P^EN` | `χ^EE` | ablation isolating the descriptor from the kernel |
| D | `P^EE` | `χ^EE` | EE-Pattern (the EO-Pattern model of the thesis) |
| E | `P^EE` | shuffled `χ^EE` | negative control for D |
| F | `P^WE` | none | pure WE layer |
| G | `P^WE` | `χ^WE` | WE-Pattern |
| H | `P^WE` | shuffled `χ^WE` | negative control for G |

## Results

All five experiments are complete; the benchmark has 20/20 seeds for all
twelve models on both datasets. `RESULTS.md` holds every table with the reading of each experiment, and
`python summarise.py` regenerates the listing from the `results_*.json`
files.  `./run_all.sh` reproduces everything; each script writes a resumable
results file, so an interrupted run restarts where it stopped.

## Install

From this directory, once:

```bash
pip install -e .
```

This installs `eompp` (and its dependencies: numpy, scipy, torch) in editable
mode, so every experiment folder can `import eompp`. The Benchmark experiment
additionally needs TopoNetX:

```bash
(cd Benchmark && pip install -r requirements_benchmark.txt)
```

## Run

`./run_all.sh` runs all five in order.  Individually:

**Experiment 1 - Cardinality twins (graph classification).** The 50 % results of the clique-only models and chi-free EO variants follow from the paired construction and feature symmetry; the 100 % results of the chi-based variants are empirical outcomes of the committed run.
```bash
cd Synthetic
python experiment_synthetic.py
# variants:  --quick  |  --feature-mode random
```

**Experiment 2 - Multiplicity (node classification).** Canonical configuration:
```bash
cd Multiplicity
python multiplicity.py --n-gadgets 450 --m-decoys 4 --n-seeds 10
```

**Experiment 3 - Benchmark (node classification).** 50/25/25 random splits, multi-seed, resumable.
```bash
cd Benchmark
python experiment_benchmark.py --dataset cora     --n-seeds 20 --out reproduced_results_cora.json
python experiment_benchmark.py --dataset citeseer --n-seeds 20 --out reproduced_results_citeseer.json
# quick smoke test:  python experiment_benchmark.py --dataset cora --quick
```

**Experiment 4 - Complementarity, EE factorization (P/Q/R gadget).** Canonical configuration:
```bash
cd Complementarity
python pqr.py --n-gadgets 450 --m-decoys 4 --n-seeds 10 --n-layers 1 --out reproduced_results_complementarity.json
```

**Experiment 5 - Complementarity, WE factorization.** The mirror gadget, at both dilutions:
```bash
cd Complementarity
python pqr_we.py --n-gadgets 450 --m-decoys 4 --n-seeds 10
python pqr_we.py --n-gadgets 450 --m-decoys 2 --n-seeds 10 \
    --out results_complementarity_we_m2.json
```

Each folder has its own README with the experiment design, the results table,
and per-experiment flags.

## Repository structure

```
.
├── eompp/                          shared method (installable package)
│   ├── eo.py                       P^EN, P^EE, P^WE, χ^EE, χ^WE, incidence helpers
│   ├── layers.py                   GCNLayer, GINLayer, AllDeepSetsLayer, RWPatternLayer
│   ├── node_models.py              twelve node-classification models + registry
│   ├── node_training.py            train/eval loop, tensor conversion, chi shuffle
│   ├── gadgets.py                  target-only split + shared driver for exp. 2, 4, 5
│   └── metrics.py                  macro_f1
├── pyproject.toml
├── run_all.sh                      reproduces all five experiments
├── summarise.py                    prints every results_*.json as a table
├── RESULTS.md                      all tables with the reading of each experiment
├── Synthetic/                      Experiment 1 - cardinality twins
│   ├── data_synthetic.py           twin-pair generator and split
│   ├── models_synthetic.py         graph-classification wrappers (pooling + head)
│   └── experiment_synthetic.py     training harness and results table
├── Multiplicity/                   Experiment 2 - multiplicity gadget
│   └── multiplicity.py             3-uniform generator, signature check, driver
├── Benchmark/                      Experiment 3 - co-citation node classification
│   ├── data/                       committed datasets (Cora, Citeseer)
│   ├── data_benchmark.py           data loading, TopoNetX incidence, split
│   └── experiment_benchmark.py     multi-seed driver (resumable)
└── Complementarity/                Experiments 4 and 5 - P/Q/R gadgets
    ├── pqr.py                      EE-targeted gadget generator and driver
    └── pqr_we.py                   WE-targeted mirror gadget and driver
```