# CASPer

This repository contains the verified data, reproducible analysis code, tests,
frozen benchmark results, and regenerated figures for:

**Site-Conditioned Edit Chemistry for Cyclic Peptide Permeability Modeling**

## Repository layout

- `data/` — corrected 7,224-sample dataset and fixed random/cluster splits.
- `src/` — minimal production code required by the retained pipelines and tests.
- `configs/benchmark/primary_ablation.yaml` — final 50-trial primary protocol.
- `scripts/` — final experiment, statistical-analysis, freeze, and plotting scripts.
- `tests/` — validated test suite.
- `results/final_experiments/` — frozen final tables, raw runs, predictions,
  bootstraps, SHAP, scaffold, and time-forward outputs.
- `results/final_experiments/figures/` — regenerated PNG/PDF figures.
- `results/final_experiments/summary_tables/` — final human-readable tables.

## Verified completion

- Primary descriptor ablation: 70/70 seed-level runs.
- Estimator × descriptor matrix: 175/175 seed-level runs.
- Five-seed SHAP: complete.
- Time-forward diagnostics: eight cutoffs complete.
- Scaffold ranking: 49 peptide families complete.
- Tests: 28 passed, 0 failed.

The canonical machine-readable freeze is:

`results/final_experiments/FINAL_RESULTS_FREEZE.json`

## Reproduce figures

```bash
.venv/bin/python scripts/generate_figures.py
```

Final PNG/PDF figures and their source data are written to:

`results/final_experiments/figures/`

## Validation

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile scripts/*.py
```
