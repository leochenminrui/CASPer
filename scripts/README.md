# Scripts

All scripts should be run from the **project root** (not from inside `scripts/`).

## Data Preparation

| Script | Command | Output |
|--------|---------|--------|
| `convert_to_pem_schema.py` | `python scripts/convert_to_pem_schema.py --dataset cycpeptmpdb` | `data/processed/pem_schema/cycpeptmpdb_pampa.{jsonl,parquet}` |
| `01_preprocess_cycpeptmpdb.py` | `python scripts/01_preprocess_cycpeptmpdb.py` | `data/raw/cycpeptmpdb_pampa.csv` |
| `02_generate_splits.py` | `python scripts/02_generate_splits.py` | `data/splits/CycPeptMPDB_PAMPA/{random,sequence_cluster,...}/` |

## Audit & Census

| Script | Command | Output |
|--------|---------|--------|
| `run_full_audit.py` | `python scripts/run_full_audit.py` | `results/benchmark/audit/{per_sample_audit,excluded_samples,monomer_coverage}.csv`, `audit_summary.json` |
| `run_strict_census.py` | `python scripts/run_strict_census.py` | Strict anchor-resolvability census |
| `run_census.py` | `python scripts/run_census.py` | Initial dataset census |

## Main Benchmarks

| Script | Command | Output |
|--------|---------|--------|
| `run_benchmark.py` | `python scripts/run_benchmark.py --quick` (smoke) / `--full` (production) | `results/benchmark/{random,sequence_cluster}/seed_*/`, `results/benchmark/summary/` |
| `run_estimator_comparison.py` | `python scripts/run_estimator_comparison.py` | `results/benchmark/estimator_comparison/comparison_summary.csv` |

## Analysis

| Script | Command | Output |
|--------|---------|--------|
| `run_shap_analysis.py` | `python scripts/run_shap_analysis.py` | `results/benchmark/feature_importance/{shap_summary_*,shap_top20_*,shap_group_contribution_*}.{pdf,csv}` |
| `run_scaffold_ranking.py` | `python scripts/run_scaffold_ranking.py` | `results/benchmark/{time_forward_ranking,scaffold_ranking}/` |

## Supplementary Controls

| Script | Command | Output |
|--------|---------|--------|
| `run_all_supplement_5seed.py` | `python scripts/run_all_supplement_5seed.py` | `results/supplement_5seed/` |
| `run_perturbation_seeds.py` | `python scripts/run_perturbation_seeds.py` | `results/perturbation/` |

## Statistical Analysis

| Script | Command | Output |
|--------|---------|--------|
| `compute_paired_significance_tests.py` | `python scripts/compute_paired_significance_tests.py` | Statistical test results |
| `compute_bootstrap_confidence_intervals.py` | `python scripts/compute_bootstrap_confidence_intervals.py` | Bootstrap 95% CIs |

## Figure Generation

| Script | Command | Output |
|--------|---------|--------|
| `make_revision_figures.py` | `python scripts/make_revision_figures.py` | `figures/revision/{fig2-6,figS1-S4}.png` |
| `generate_publication_figures.py` | `python scripts/generate_publication_figures.py` | `figures/Figure_{1-4}_*.{pdf,png}` (original) |
| `make_supplementary_table_S1.py` | `python scripts/make_supplementary_table_S1.py` | Supplementary Table S1 |

## Utilities

| Script | Purpose |
|--------|---------|
| `00_verify_setup.py` | Check environment and directory structure |
| `00_validate_splitting.py` | Verify no train/test leakage in splits |
| `convert_to_pem_schema.py` | Convert raw CSV to unified PEM schema |
| `generate_chem_repr.py` | Generate chemistry descriptor representations |
| `split_summary_report.py` | Print split statistics |
| `baseline_summary.py` | Print aggregated baseline results |
| `eval_baseline.py` | Evaluate a single trained baseline |
| `eval_harder_split.py` | Evaluate models on sequence-cluster split |
| `create_sequence_cluster_split.py` | Create 70% identity sequence-cluster split |
| `create_stricter_sequence_cluster_split.py` | Create 50% identity stricter split |
| `compile_all_results.py` | Compile all per-seed JSON results into `compiled_metrics.json` |
| `compile_final_summary.py` | Final summary aggregation |

## Quick Start (Smoke Test, ~15 min)

```bash
python scripts/run_full_audit.py
python scripts/run_benchmark.py --quick
python scripts/run_benchmark.py --summary-only
python scripts/run_estimator_comparison.py
python scripts/run_shap_analysis.py
python scripts/run_scaffold_ranking.py
python scripts/make_revision_figures.py
```

## Full Reproduction (~2-6 hours)

```bash
# 1. Data preparation
python scripts/convert_to_pem_schema.py --dataset cycpeptmpdb
python scripts/02_generate_splits.py

# 2. Full audit
python scripts/run_full_audit.py

# 3. Main benchmarks
python scripts/run_benchmark.py --full

# 4. Estimator comparison
python scripts/run_estimator_comparison.py

# 5. SHAP analysis
python scripts/run_shap_analysis.py

# 6. Ranking simulations
python scripts/run_scaffold_ranking.py

# 7. Supplementary controls
python scripts/run_all_supplement_5seed.py

# 8. Statistical analysis
python scripts/compute_bootstrap_confidence_intervals.py
python scripts/compute_paired_significance_tests.py

# 9. Figures
python scripts/make_revision_figures.py
```

## Legacy Scripts

Old pipeline scripts (numbered 03–08, 00_check/verify) have been removed.
They were superseded by the unified benchmark framework (`run_benchmark.py`).
