# Reproducibility Notes

This file records an independent reproduction check of the committed artifacts,
performed in a clean `uv`-managed Python 3.12 environment installed from
`requirements.txt`. It documents what reproduces exactly, what does not, and the
fixes applied in this branch.

## What was verified to reproduce ✅

| Manuscript item | Script | Result |
|---|---|---|
| Table 1 — dataset census | `scripts/run_full_audit.py` | **Byte-identical** to committed `results/benchmark/audit/` (7,298 → 7,224 retained, 74 excluded, 98.99%). |
| Table 3 / benchmark summary | `scripts/run_benchmark.py --summary-only` | `benchmark_table_for_manuscript.csv` and `mean_std_by_model.csv` regenerate **byte-identical** from the committed per-seed `metrics.json`. |

The core point estimates and the dataset audit are therefore reproducible from the
shipped code + data, and they match the manuscript.

## Issues found and fixed in this branch 🔧

1. **Missing dependency `pydantic`.** `src/data/pem_schema.py` imports
   `pydantic` (v2 API: `field_validator`, `model_validator`), but it was absent
   from `requirements.txt`, so a fresh install could not import `src.benchmark`.
   Added `pydantic>=2.0.0`.

2. **README headline numbers were stale.** The §1 "Key result" table and the
   in-text p-values quoted a superseded result generation
   (`Anchor-aware = 0.4717…`, still present in
   `figures/source_data/Figure_4_source_data.csv`). They have been corrected to
   match manuscript Table 3 and `benchmark_table_for_manuscript.csv`
   (full A+B+C R² = 0.4665 ± 0.0056, chemistry 0.4253, sequence 0.2574).

3. **Documentation fixes:** wrong clone dir (`cd CASPer_re` → `cd CASPer`);
   model-registry count (`20+ models, 7 roles` → `15 models, 4 roles`, matching
   `src/benchmark/registry.py`); undeclared **MMseqs2/CD-HIT** requirement for the
   cluster split is now documented; citation/hardware placeholders made explicit.

## Open issue requiring author action ⚠️ (NOT changed here)

**`scripts/compute_paired_significance_tests.py` cannot run on the shipped repo**
and the statistics it produced (Supplementary Table S3) are from the superseded
generation:

- The script hard-codes paths from an old layout that no longer exists, e.g.
  `results/anchor_descriptor_xgb/test_results_seed42.json` and
  `results/baselines/.../descriptor_only_xgboost/test_results_seed0.json`.
- It compares a single representative seed (seed 42 for the anchor model vs seed 0
  for the descriptor model), not the 5-seed aggregate reported in Table 3.
- Consequently `supplement/Supplementary_Table_S3_statistical_comparisons.csv`
  still reports `0.4717 / 0.4327 / 0.2635` with `n_anchor = 3` seeds — inconsistent
  with the corrected 5-seed Table 3.

Recomputed on the committed 5-seed `metrics.json` (paired by seed, scipy verified):

| Comparison | S3 / manuscript (old) | Corrected (5-seed) |
|---|---|---|
| Chemistry vs Sequence | p = 2.9 × 10⁻⁸ | p = 5.1 × 10⁻⁶ paired / 1.8 × 10⁻⁸ Welch |
| Site-conditioned A+B+C vs Chemistry | p = 2.6 × 10⁻⁶ (Welch, n=3 vs 5) | p = 1.1 × 10⁻⁴ |
| A+B vs Chemistry | (not separately reported) | p = 0.14 — not significant at α = 0.05 |

The qualitative conclusions hold, but `compute_paired_significance_tests.py` should
be ported to the current `results/benchmark/random/seed_*/{model}/` layout (the
per-seed `predictions.csv` needed for per-observation Wilcoxon tests **are**
shipped), S3 regenerated, and the manuscript p-values + the 3-vs-5-seed count
reconciled. This is left to the authors because it changes published statistics.

## Environment caveat

`requirements.txt` pins only lower bounds (`>=`). The reproduction above resolved
to newer releases than the paper likely used (e.g. scikit-learn 1.9, xgboost 3.3).
The aggregation/audit steps are unaffected, but exact re-training of models may
drift; consider pinning upper bounds or shipping a lockfile for archival
reproducibility.
