# Experiment 2 - Multiplicity

The mirror image of the cardinality twin experiment in `../Synthetic/`.

|                          | `Synthetic/` (experiment 1) | `Multiplicity/` (experiment 2) |
|--------------------------|-----------------------------|--------------------------------|
| held fixed               | adjacency, multiplicity     | adjacency, cardinality         |
| varied                   | hyperedge cardinality       | co-occurrence multiplicity     |
| provably blind           | every kernel (constant features) | both descriptors (`r`-uniformity) |
| expected to separate     | `χ^EE`, `χ^WE`              | `P^EE`, `P^WE`                 |

## Construction

Every hyperedge has size 3, so the hypergraph is 3-uniform.  For each target
node `v`:

* one informative neighbour `uQ`, connected through **two** size-3 hyperedges
  (`c_vu = 2`), whose feature carries the true class of `v`;
* `m` decoys, each connected through **one** size-3 hyperedge (`c_vu = 1`),
  with random classes;
* neutral filler vertices completing the hyperedges.

All neighbours are adjacent to `v` and share the same hyperedge cardinality,
so the informative one is identifiable only from how many hyperedges it
shares with `v`.

## Why the experiment is diagnostic by construction

3-uniformity forces two exact identities, both asserted in
`check_signatures`:

1. `χ^EE_vu = χ^WE_vu = c_3` for every adjacent pair, so both descriptors are
   constant and provably carry no information (structural-inertness
   proposition).
2. `Z^EE_vu = c_vu/(r-1)` and `s_H(v) = (r-1) d_H(v)`, hence
   `P^EE(v,u) = c_vu / ((r-1) d_H(v)) = P^WE(v,u)`: the two kernels coincide.

Consequently the run must satisfy `B == F` and `D == E == G == H` exactly.
These are not approximate agreements: under 3-uniformity those variants are
literally the same model, so any deviation indicates a bug.

## Run

```bash
python multiplicity.py                       # 450 gadgets, m=4, 10 seeds
python multiplicity.py --n-gadgets 200 --n-seeds 3
```
