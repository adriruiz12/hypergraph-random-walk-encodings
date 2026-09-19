"""
experiment_benchmark.py
======================

Main entry point for the realistic node-classification benchmark
(Cora, Citeseer co-citation hypergraphs).

The twelve models and the train/eval routines live in the shared
`eompp` package; this module only holds the benchmark-specific concerns:
the configuration, the per-dataset multi-seed driver with resumable
checkpointing, and the CLI.

Run
---
    python experiment_benchmark.py                  # all datasets
    python experiment_benchmark.py --dataset cora   # one dataset
    python experiment_benchmark.py --quick          # fast sanity check
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from data_benchmark import load_dataset, make_split, DATASETS
from eompp.node_models import build_model, MODEL_SPECS
from eompp.node_training import to_tensors, train_one, evaluate, shuffled_chi


# --------------------------------------------------------------------------
# Configuration.
# --------------------------------------------------------------------------
class Config:
    def __init__(self):
        self.datasets = DATASETS
        self.max_card = None      # K := max_e |e|, computed per dataset
        self.use_toponetx = True
        self.hidden = 128
        self.n_layers = 2
        self.dropout = 0.5
        self.lr = 1e-3
        self.weight_decay = 5e-4
        self.max_epochs = 200
        self.patience = 40
        self.seeds = tuple(range(20))   # 20 random 50/25/25 splits (AllSet convention); reduce with --n-seeds
        self.results_path = "results_benchmark.json"


# --------------------------------------------------------------------------
# One dataset.
# --------------------------------------------------------------------------
def run_dataset(dataset, cfg, device, model_names=None, prev=None,
                save_cb=None):
    """Run a (subset of) models on one dataset, with resumable checkpointing.
    
    `model_names` selects which models to run (default: all).
    `prev` is an existing results dict for this dataset to merge into.
    `save_cb(results)` is called after every model so that partial
    progress survives even if the process is killed mid-run.
    """

    print("=" * 74)
    print(f"DATASET: {dataset}")
    print("=" * 74)

    specs = [(n, k) for n, k in MODEL_SPECS
             if model_names is None or n in model_names]

    data = load_dataset(dataset, max_card=cfg.max_card,
                        use_toponetx=cfg.use_toponetx)
    batch = to_tensors(data, device)

    active_mask = torch.tensor(data["d_h"] > 0, dtype=torch.bool).to(device)

    n_classes = data["n_classes"]
    print(f"nodes={data['n_nodes']}  features={data['n_features']}  "
          f"classes={n_classes}  hyperedges={data['incidence'].shape[1]}  "
          f"clique-expansion edges={data['edge_index'].shape[1]}  "
          f"K={data['chi_dim'] + 1}")

    results = dict(prev) if prev else {}
    results["_meta"] = dict(max_card=data["chi_dim"] + 1)  # resolved K

    for name, shuffle_key in specs:
        # resume: reuse per-seed scores already on disk for this model
        cached = results.get(name, {})
        accs = list(cached.get("acc_per_seed", []))
        f1s = list(cached.get("f1_per_seed", []))
        accs_act = list(cached.get("acc_act_per_seed", []))
        done = len(accs)
        if done >= len(cfg.seeds):
            print(f"  skip: {name:<26s}  (already has {done} seeds)")
            continue

        for seed in cfg.seeds[done:]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            tr, va, te = make_split(data["y"], seed=seed)
            train_mask = torch.zeros(data["n_nodes"], dtype=torch.bool)
            val_mask = torch.zeros(data["n_nodes"], dtype=torch.bool)
            test_mask = torch.zeros(data["n_nodes"], dtype=torch.bool)
            train_mask[tr] = val_mask[va] = test_mask[te] = True
            train_mask = train_mask.to(device)
            val_mask = val_mask.to(device)
            test_mask = test_mask.to(device)
            active_test = test_mask & active_mask

            b = (shuffled_chi(batch, shuffle_key, seed)
                 if shuffle_key else batch)

            model = build_model(
                name, n_features=data["n_features"], chi_dim=data["chi_dim"],
                n_classes=n_classes, hidden=cfg.hidden,
                n_layers=cfg.n_layers, dropout=cfg.dropout).to(device)
            model = train_one(model, b, train_mask, val_mask, n_classes, cfg)
            acc, f1 = evaluate(model, b, test_mask, n_classes)
            acc_act, _ = evaluate(model, b, active_test, n_classes)
            accs.append(acc)
            f1s.append(f1)
            accs_act.append(acc_act)

            # persist after every seed so a kill never wastes work
            results[name] = dict(
                acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)),
                acc_per_seed=accs, f1_per_seed=f1s, acc_act_mean=float(np.mean(accs_act)),
                acc_act_std=float(np.std(accs_act)), acc_act_per_seed=accs_act)
            if save_cb is not None:
                save_cb(results)

        print(f"  done: {name:<26s}  "
              f"acc = {results[name]['acc_mean'] * 100:5.1f} "
              f"+/- {results[name]['acc_std'] * 100:.1f}  "
              f"(active: {results[name]['acc_act_mean'] * 100:5.1f})")
        if save_cb is not None:
            save_cb(results)              # persist after every model

    # results table (only rows that have results so far)
    print()
    print(f"{'Model':<28s} {'Accuracy (%)':>15s} {'Macro-F1 (%)':>15s}")
    print("-" * 74)
    for name, _ in MODEL_SPECS:
        if name not in results:
            continue
        r = results[name]
        print(f"{name:<28s} "
              f"{r['acc_mean'] * 100:7.1f} +/- {r['acc_std'] * 100:4.1f} "
              f"{r['f1_mean'] * 100:7.1f} +/- {r['f1_std'] * 100:4.1f}")
    print("-" * 74)
    print()
    return results


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"],
                        default="all")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cpu", action="store_true",
                        help="force CPU even if CUDA is available")
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="use seeds 0..n-1 instead of the default 20")
    parser.add_argument("--models", type=str, default=None,
                        help="comma-separated subset of model names to run")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.dataset != "all":
        cfg.datasets = (args.dataset,)
    if args.quick:
        cfg.seeds = (0, 1)
        cfg.max_epochs = 120
        cfg.results_path = "results_benchmark_quick.json"
    if args.hidden is not None:
        cfg.hidden = args.hidden
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.patience is not None:
        cfg.patience = args.patience
    if args.lr is not None:
        cfg.lr = args.lr
    if args.n_seeds is not None:
        cfg.seeds = tuple(range(args.n_seeds))
    if args.out is not None:
        cfg.results_path = args.out
    model_names = (set(m.strip() for m in args.models.split(","))
                   if args.models else None)

    device = torch.device("cpu")
    print(f"device: {device}   seeds: {cfg.seeds}   "
          f"hidden: {cfg.hidden}   layers: {cfg.n_layers}\n")

    # merge into an existing results file if present
    if os.path.exists(cfg.results_path):
        with open(cfg.results_path) as fh:
            all_results = json.load(fh).get("results", {})
    else:
        all_results = {}

    for ds in cfg.datasets:
        prev = all_results.get(ds)

        def save_cb(partial, _ds=ds):
            """Persist this dataset's partial results to disk."""

            all_results[_ds] = partial
            with open(cfg.results_path, "w") as fh:
                json.dump(dict(config=vars(cfg),
                               results=all_results), fh, indent=2)

        all_results[ds] = run_dataset(ds, cfg, device,
                                      model_names=model_names, prev=prev,
                                      save_cb=save_cb)
        save_cb(all_results[ds])

    print(f"saved -> {cfg.results_path}")


if __name__ == "__main__":
    sys.exit(main())
