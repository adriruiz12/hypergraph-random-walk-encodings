#!/usr/bin/env bash
# Reproduce every experiment.  Each script writes a resumable results_*.json,
# so an interrupted run can be restarted with the same command.
set -e
pip install -e .

echo "== 1. Cardinality twins (graph classification) =="
( cd Synthetic      && python experiment_synthetic.py )

echo "== 2. Multiplicity gadget (node classification) =="
( cd Multiplicity   && python multiplicity.py )

echo "== 3. Co-citation benchmark (node classification) =="
( cd Benchmark      && python experiment_benchmark.py --cpu )

echo "== 4. Complementarity, EE factorization =="
( cd Complementarity && python pqr.py )

echo "== 5. Complementarity, WE factorization =="
( cd Complementarity && python pqr_we.py )
( cd Complementarity && python pqr_we.py --m-decoys 2 \
      --out results_complementarity_we_m2.json )   # dilution control
