# Revision changelog

## Repository and pipeline

- Created protected branch `minor-revision-complete`.
- Consolidated every protocol-compatible raw run under `results/minor_revision_experiments/raw_runs/`, then removed superseded result trees, figures, scripts, documentation, and duplicate submission assets.
- Retained only the corrected data, fixed splits, required production code, final experiment/statistical/plotting scripts, tests, frozen results, and final submission materials.
- Repaired classical-ML loading so descriptor experiments do not require optional PyTorch wrappers.
- Corrected ablation feature-name projection so names match matrix columns.
- Updated obsolete split-test fixtures to comply with the production sample-ID and amino-acid schemas.
- Added revision-specific validation tests for dataset size, split isolation, descriptor membership, prediction alignment, SHAP grouping, metrics, seeds, test leakage, and checkpoint validation.

## Experiments

- Completed all missing 50-trial primary descriptor-ablation cells.
- Reran the five random Chemistry + Context cells because committed supplementary metrics lacked predictions and could not be reproduced from saved parameters in the resolved environment.
- Completed all missing 10-trial estimator-by-descriptor cells.
- Recomputed five-seed TreeSHAP attribution using XGBoost native TreeSHAP contributions.
- Recomputed all eight time-forward cutoffs with compound predictions, ECFP similarity diagnostics, grouped confidence intervals, and source/shift diagnostics.
- Recomputed the 49-family scaffold-ranking analysis with exact valid-pair counts and family bootstrap intervals.

## Manuscript/source status

No main-manuscript, Supplementary Information, response-letter, or original bibliography source exists in the repository or its reachable Git history. A filesystem search under `/home/minrui` found only `figures/revision/figure_latex_snippets.tex`. Therefore manuscript editing, page/line locations, and PDF compilation cannot be performed without external source files. No manuscript was reconstructed from a PDF.
