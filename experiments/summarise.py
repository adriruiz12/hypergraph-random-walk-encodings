"""Print every results_*.json in the repository as one table per experiment."""

import glob
import json
import os

ORDER = ["MLP", "Clique-GCN", "Clique-GIN", "AllDeepSets",
         "A: EN", "B: EE", "C: EN + chi^EE", "D: EE-Pattern",
         "E: EE-Pattern, shuffled chi", "F: WE", "G: WE-Pattern",
         "H: WE-Pattern, shuffled chi"]


def rows(res):
    """Yield (name, mean, std, n) for a flat results dict.

    Non-model entries (e.g. "_meta", holding the resolved K) are skipped.
    """
    for name in ORDER + [k for k in res if k not in ORDER]:
        r = res.get(name)
        if r and "acc_mean" in r:
            yield name, r["acc_mean"], r["acc_std"], len(r["acc_per_seed"])


for path in sorted(glob.glob("*/results_*.json")):
    res = json.load(open(path))["results"]
    nested = all(isinstance(v, dict) and "acc_mean" not in v
                 for v in res.values())
    blocks = res.items() if nested else [(os.path.dirname(path), res)]
    for title, block in blocks:
        print(f"\n=== {path}  [{title}] ===")
        print(f"{'Model':<30s}{'Acc (%)':>10s}{'std':>7s}{'seeds':>7s}")
        for name, m, s, n in rows(block):
            print(f"{name:<30s}{m * 100:10.1f}{s * 100:7.1f}{n:7d}")
