# eompp

**Three random-walk-induced, vertex-centered message-passing paradigms on hypergraphs**

This repository contains the surveys, implementation, and experiments developed as part of a
BSc thesis project and its subsequent extension. The project studies how pairwise,
vertex-centered message passing can be conditioned directly by hypergraph connectivity
patterns, rather than relying only on an unweighted graph projection or on a fully general
relational structure.

Three random-walk kernels are placed within a common formal framework:

| Paradigm | Kernel | Descriptor |
|---|---|---|
| **EN** (equal-nodes)    | `P^EN(v,u) = 1/\|N_H(v)\|`     | none (EN mediates no hyperedge) |
| **EE** (equal-edges)    | `P^EE(v,u) = Z^EE_vu / d_H(v)` | `χ^EE` (1/(\|e\|-1)-weighted)   |
| **WE** (weighted-edges) | `P^WE(v,u) = c_vu / s_H(v)`    | `χ^WE` (uniformly weighted)     |

The **EE-Pattern** layer (`P^EE` + `χ^EE`) is the **EO-Pattern** model of the submitted
bachelor's thesis; **EN** is the uniform-aggregation baseline used there; **WE** and
**WE-Pattern** are new, formalized and evaluated in `docs/Random_Walk_Propagations.pdf`.

## Guiding question

> Can a pairwise, vertex-centered message-passing layer be conditioned directly by hyperedge
> connectivity patterns (Eidi & Otter, 2025), rather than relying only on an unweighted graph
> projection or on a general relational structure (Taha et al., 2025)? And, once EE is placed
> inside a family of such kernels, is EE's cardinality information genuinely distinct from the
> multiplicity information a different walk (WE) can extract, or are they two factorizations of
> the same underlying pairwise structure?

## Repository structure

```text
.
├── experiments/
│   ├── Benchmark/              # Cora and Citeseer co-citation benchmarks
│   ├── Complementarity/        # P/Q/R complementarity gadgets (EE- and WE-targeted)
│   ├── Multiplicity/           # 3-uniform multiplicity-separation gadget
│   ├── Synthetic/              # Cardinality separation beyond clique expansion
│   ├── eompp/                  # Shared implementation of the three kernels/descriptors and baselines
│   ├── RESULTS.md              # All result tables with the reading of each experiment
│   ├── run_all.sh              # Reproduces all five experiments
│   ├── summarise.py            # Regenerates the results listing from results_*.json
│   ├── pyproject.toml
│   └── README.md               # Detailed experimental documentation
├── docs/
│   ├── Random_Walk_Propagations.pdf   # EN/EE/WE note: formalization + five experiments
│   ├── Survey_of_Random_Walk_Variants_on_Graphs_and_Their_Natural_Lifts_to_Higher_Order_Structures.pdf
│   ├── Survey_of_higher_order_structures_used_in_graph_learning_and_the_associated_message_passing_frameworks.pdf
│   └── Updated_TFG_Matemáticas_Public.pdf
└── README.md
```

Each experiment directory contains its own README, executable scripts, dependency file, and
committed JSON results.

## The three kernels and descriptors

Let `H = (V, E)` be a hypergraph, let `E(v,u)` be the collection of hyperedges containing both
`v` and `u` (repeated hyperedges counted with multiplicity), let `d_H(v)` be the number of
hyperedges of cardinality at least two incident to `v`, and let `c_vu = |E(v,u)|` be the
co-occurrence multiplicity. Define

```math
Z^{\mathrm{EE}}_{vu} = \sum_{e \in \mathcal{E}(v,u)} \frac{1}{|e|-1},
\qquad
s_H(v) = \sum_{e \in \mathcal{E}(v)} (|e|-1).
```

The three transition kernels are

```math
P^{\mathrm{EN}}(v,u) = \frac{1}{|\mathcal{N}_H(v)|},
\qquad
P^{\mathrm{EE}}(v,u) = \frac{Z^{\mathrm{EE}}_{vu}}{d_H(v)},
\qquad
P^{\mathrm{WE}}(v,u) = \frac{c_{vu}}{s_H(v)},
```

all row-stochastic. EE weights each shared hyperedge by the reciprocal of its size; WE weights
every shared hyperedge equally, so it retains multiplicity but discards individual cardinality;
EN retains only adjacency. The two structural descriptors are the conditional (posterior)
distribution of shared-hyperedge cardinality under each walk:

```math
\chi^{\mathrm{EE}}_{vu} = \frac{1}{Z^{\mathrm{EE}}_{vu}} \sum_{e \in \mathcal{E}(v,u)} \frac{1}{|e|-1}\, \mathbf{e}_{b(|e|)},
\qquad
\chi^{\mathrm{WE}}_{vu} = \frac{1}{c_{vu}} \sum_{e \in \mathcal{E}(v,u)} \mathbf{e}_{b(|e|)},
```

fixed structural descriptors, not learned embeddings; the message function learns how to use
them. One Pattern layer performs

```math
m_v^{(\ell+1)} = \sum_{u \in \mathcal{N}_H(v)} P^X(v,u)\, \psi^{(\ell)}_X\!\left(h_v^{(\ell)}, h_u^{(\ell)}, \chi^X_{vu}\right),
\qquad
h_v^{(\ell+1)} = \varphi^{(\ell)}_X\!\left(h_v^{(\ell)}, m_v^{(\ell+1)}\right),
```

for `X ∈ {EE, WE}` (the pure EN, EE, WE layers omit `χ`). `docs/Random_Walk_Propagations.pdf`
proves that, at the pair level and once the cardinality resolution `K ≥ max_e |e|` is exact,
`(Z^EE_vu, χ^EE_vu)` and `(c_vu, χ^WE_vu)` are mutually invertible: EE-Pattern and WE-Pattern
carry the same pair information, differing only in vertex normalization (`d_H` vs `s_H`) and in
how that information is split between the aggregation weight and the message feature. This is
why the experiments below are organized as mirror pairs.

The implementation in `experiments/eompp/` uses sparse incidence matrices to compute all three
kernels and both descriptors, and pure PyTorch for the message-passing layers.

## Models and ablations

The experiments compare the three paradigms against feature-only, graph-based, and
incidence-based baselines, using an eight-variant ablation ladder (defined once in
`eompp.node_models.RW_SPECS`):

| Variant | Kernel | Descriptor | Role |
|---|---|---|---|
| A | `P^EN` | none | pure EN layer (uniform baseline of the thesis) |
| B | `P^EE` | none | pure EE layer |
| C | `P^EN` | `χ^EE` | descriptor isolated from the EE kernel |
| **D** | `P^EE` | `χ^EE` | **EE-Pattern (the submitted EO-Pattern model)** |
| E | `P^EE` | shuffled `χ^EE` | negative control for D |
| F | `P^WE` | none | pure WE layer |
| **G** | `P^WE` | `χ^WE` | **WE-Pattern** |
| H | `P^WE` | shuffled `χ^WE` | negative control for G |

Additional baselines: MLP (features only), Clique-GCN, Clique-GIN, and an in-house
mean-normalized AllDeepSets-style incidence baseline.

## Experiments and results

Five experiments, organized in mirror pairs, each isolating one axis of hypergraph structure.
Full tables and per-experiment readings are in
[`experiments/RESULTS.md`](experiments/RESULTS.md); summary below.

### 1-2. Cardinality vs. multiplicity separation

Twin-hypergraph classification (`Synthetic/`) isolates cardinality at fixed multiplicity; the
3-uniform gadget (`Multiplicity/`) isolates multiplicity at fixed cardinality. Each axis is
detected only by the ingredient built to detect it:

| Experiment | Kernels alone | Descriptors alone |
|---|---:|---:|
| 1. Cardinality twins | 50.0% (chance) | **100.0%** |
| 2. Multiplicity gadget | **86.8%** (EE = WE) | 70.4% ≈ EN (inert) |

### 3. Co-citation benchmark (Cora, Citeseer)

All eight variants plus baselines, 20 seeds/dataset. As in the submitted thesis, the ladder is
flat: EE-Pattern and its shuffled control agree within noise, and Clique-GCN remains the
strongest single model on Cora. Structural diagnostics in `RESULTS.md` show why: on co-citation
data, `c_vu = 1` for ~80% of adjacent pairs, so all three kernels collapse onto (almost) the
same object before the model sees the data.

### 4-5. Complementarity (P/Q/R gadgets)

The EE-targeted gadget (`Complementarity/pqr.py`) and its WE-targeted mirror
(`Complementarity/pqr_we.py`) each show that the corresponding Pattern layer is
super-additive (kernel alone + descriptor alone < combination), while the *other* paradigm's
kernel alone resolves the same task without any descriptor:

| Gadget | Kernel alone | Descriptor alone | Combined (Pattern) |
|---|---:|---:|---:|
| 4. EE-targeted | +5.5 | −1.6 | **+17.4** |
| 5. WE-targeted (matched dilution) | +4.6 | ≈0 | **+20.6** |

This demonstrates that complementarity is a property of the chosen factorization, not of the
hypergraph information itself (see Proposition 9.2 in the note).

## Installation

Python 3.9 or later is required.

```bash
cd experiments
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The benchmark additionally requires TopoNetX:

```bash
(cd Benchmark && pip install -r requirements_benchmark.txt)
```

## Reproducing the experiments

```bash
cd experiments
./run_all.sh          # runs all five experiments in order
python summarise.py   # regenerates the results listing from results_*.json
```

See `experiments/README.md` for per-experiment commands and flags.

## Background documents

- **`docs/Random_Walk_Propagations.pdf`**: formalizes EN, EE, and WE as instances of one
  message-passing template, introduces the WE-conditioned descriptor `χ^WE`, proves
  row-stochasticity and the pairwise information equivalence of the two Pattern layers, and
  reports all five experiments above. It extends the submitted work without modifying the
  thesis or the presentation delivered at UAB.
- **Higher-order structures in graph learning:** hypergraphs, simplicial, cellular, and
  combinatorial complexes; message-passing frameworks; expressivity; oversmoothing;
  oversquashing; intrinsic versus relational formulations.
- **Random-walk variants and higher-order lifts:** Markov-chain foundations, graph random
  walks, and their extensions to hypergraphs and other higher-order domains.

## References

- Eidi, M., & Otter, N. (2025). *Geometric characterisation of structural and regular
  equivalences in undirected (hyper)graphs*. arXiv:2512.24961.
- Taha, D., Chapman, J., Eidi, M., Devriendt, K., & Montúfar, G. (2025). *Demystifying
  topological message-passing with relational structures: A case study on oversquashing in
  simplicial message-passing*. ICLR 2025. arXiv:2506.06582.
- Coupette, C., Dalleiger, S., & Rieck, B. (2023). *Ollivier-Ricci curvature for hypergraphs: A
  unified framework*. ICLR 2023.
- Chien, E., Pan, C., Peng, J., & Milenkovic, O. (2022). *You are AllSet: A Multiset Function
  Framework for Hypergraph Neural Networks*. ICLR 2022.
- Carletti, T., Battiston, F., Cencetti, G., & Fanelli, D. (2020). *Random walks on
  hypergraphs*. Physical Review E, 101(2).
- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). *Neural
  message passing for quantum chemistry*. ICML 2017.
- Yadati, N., Nimishakavi, M., Yadav, P., Nitin, V., Louis, A., & Talukdar, P. (2019).
  *HyperGCN: A new method for training graph convolutional networks on hypergraphs*.
  NeurIPS 2019.