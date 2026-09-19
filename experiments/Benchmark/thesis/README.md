# Results as submitted in the bachelor's thesis

The node-classification results reported in the submitted thesis are kept
here unchanged for reference, in `results_cora.json` and
`results_citeseer.json`.  (Those two files are excluded from the packaged
archive of this repository; they live in the thesis repository.)
They use the old model names (`EO-A` ... `EO-E`), which correspond to
variants A to E of the current ladder:

| old | new |
|---|---|
| `EO-A: uniform EO baseline` | `A: EN` |
| `EO-B: P^EE only`           | `B: EE` |
| `EO-C: chi only`            | `C: EN + chi^EE` |
| `EO-D: EO-Pattern (full)`   | `D: EE-Pattern` |
| `EO-E: shuffled chi`        | `E: EE-Pattern, shuffled chi` |

They were produced before the edge list was canonicalised (see the comment in
`eompp/eo.py`), so they differ from the current `../results_benchmark.json`
by a few tenths of a point.  The differences are well inside the seed
standard deviation; the qualitative conclusions are unchanged.
