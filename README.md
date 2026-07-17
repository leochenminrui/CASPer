# CASPer — Final Minor-Revision Repository

This branch contains only the verified final data, reproducible analysis code,
tests, frozen results, regenerated figures, and revision-submission materials for:

**Site-Conditioned Edit Chemistry for Cyclic Peptide Permeability Modeling**

## Repository layout

- `data/` — corrected 7,224-sample dataset and fixed random/cluster splits.
- `src/` — minimal production code required by the retained pipelines and tests.
- `configs/benchmark/minor_revision_primary.yaml` — final 50-trial primary protocol.
- `scripts/` — final experiment, statistical-analysis, freeze, and plotting scripts.
- `tests/` — validated test suite.
- `results/minor_revision_experiments/` — frozen final tables, raw runs, predictions,
  bootstraps, SHAP, scaffold, and time-forward outputs.
- `revision_submission/` — regenerated figures, source-data tables, reports,
  bibliography, and final submission-related materials.

## Verified completion

- Primary descriptor ablation: 70/70 seed-level runs.
- Estimator × descriptor matrix: 175/175 seed-level runs.
- Five-seed SHAP: complete.
- Time-forward diagnostics: eight cutoffs complete.
- Scaffold ranking: 49 peptide families complete.
- Tests: 28 passed, 0 failed.

The canonical machine-readable freeze is:

`results/minor_revision_experiments/FINAL_RESULTS_FREEZE.json`

## Reproduce figures

```bash
.venv/bin/python scripts/generate_final_revision_figures.py
```

Final PNG/PDF figures and their source data are written to:

`revision_submission/manuscript/figures/`

## Validation

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile scripts/*.py
```

The manuscript, Supplementary Information, and response-letter LaTeX sources
were not present in the supplied repository; this blocker is documented in
`revision_submission/reports/SOURCE_FILE_BLOCKER.md`.
