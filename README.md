# CASPer: Site-Conditioned Edit Chemistry for Cyclic Peptide Permeability Modeling

> **Paper:** "Site-Conditioned Edit Chemistry for Cyclic Peptide Permeability Modeling"
>
> **Scope:** Known-site descriptor decomposition — NOT a SOTA leaderboard.

## 1. Project Overview

Cyclic peptides with non-canonical amino acid (ncAA) modifications are promising drug candidates, but predicting how specific chemical modifications affect membrane permeability requires knowing both *what* the modification is and *where* it is attached.

This repository addresses a specific scientific question:

> **Given a cyclic peptide with *known* modification sites, does incorporating site information (position, residue context, scaffold context) into chemical descriptors improve permeability prediction over chemistry alone?**

The study is a **descriptor-level contribution analysis**, not a competition for the best predictive model. We decompose prediction accuracy into three descriptor groups:

| Group | Name | Dim | What It Captures |
|-------|------|-----|------------------|
| **A** | Edit Chemistry | 10 | Whole-molecule RDKit descriptors (MW, logP, TPSA, etc.) + edit counts |
| **B** | Site-Indexed | 35 | B1: Position statistics (6) · B2: Residue composition at anchor (20) · B3: Residue properties at anchor (9) |
| **C** | Scaffold / Multi-Edit Context | 28 | Edit family distribution, type entropy, cyclization info, sequence-level features |
| **A+B+C** | Full Site-Conditioned | 73 | All features combined |

**Key result (random split, 5-seed XGBoost with Optuna HPO):**

| Model | Features | R² | RMSE | Spearman ρ |
|-------|----------|-----|------|------------|
| Sequence-only | AA composition (33 dim) | 0.2635 ± 0.0019 | 0.980 ± 0.001 | 0.521 ± 0.003 |
| Chemistry-only | Group A (10 dim) | 0.4327 ± 0.0042 | 0.860 ± 0.003 | 0.718 ± 0.003 |
| **Site-conditioned** | **Groups A+B+C (73 dim)** | **0.4717 ± 0.0019** | **0.830 ± 0.001** | **0.733 ± 0.001** |

Chemistry vs. sequence: paired *t*-test *p* = 2.9 × 10⁻⁸. Site-conditioned vs. chemistry: Welch's *t*-test *p* = 2.6 × 10⁻⁶.

On a 70% sequence-identity cluster split (novel scaffolds), chemistry descriptors retain substantial predictive power (R² ≈ 0.27), while site-aware gains diminish (ΔR² ≈ +0.002), suggesting site features are partially sequence-context-dependent.

---

## 2. Repository Structure

```
.
├── src/                              # Python package
│   ├── data/                         # Schema, loaders, converters, splitting, audit, census
│   │   ├── converters/               # Raw CSV → PEM schema converters
│   │   ├── splitting/                # Random, cluster, scaffold-aware split strategies
│   │   └── chem_repr/                # Chemical representation canonicalization
│   ├── baselines/                    # Featurizers and model classes
│   │   └── featurizers/              # anchor_aware_descriptors (Groups A/B/C), composition, ECFP, RDKit
│   ├── benchmark/                    # Unified benchmark framework
│   │   ├── registry.py               # Model registry (20+ models, 7 roles)
│   │   ├── runner.py                 # Benchmark runner (identical splits/seeds/metrics)
│   │   ├── featurizers.py            # Featurizer registry
│   │   ├── optuna_tuner.py           # Optuna hyperparameter optimization
│   │   └── evaluation.py             # Metrics computation
│   ├── evaluation/                   # Evaluation metrics
│   └── utils/                        # Logging utilities
│
├── scripts/                          # Executable pipeline scripts
│   ├── convert_to_pem_schema.py      # Raw CSV → PEM schema
│   ├── 01_preprocess_cycpeptmpdb.py  # CycPeptMPDB All → PAMPA filter
│   ├── 02_generate_splits.py         # Split generation (random + cluster)
│   ├── run_full_audit.py             # Per-sample audit (7,298 rows)
│   ├── run_benchmark.py              # Unified benchmark runner
│   ├── run_estimator_comparison.py   # Ridge/ElasticNet/RF/SVR/XGBoost comparison
│   ├── run_shap_analysis.py          # SHAP group-level attribution
│   ├── run_scaffold_ranking.py       # Time-forward + scaffold-focused ranking
│   ├── run_all_supplement_5seed.py   # Supplementary mechanistic controls
│   ├── make_revision_figures.py      # All manuscript figures (main + SI)
│   └── ...                           # Statistical tests, utilities, verification
│
├── configs/
│   ├── datasets.yaml                 # Dataset paths and settings
│   ├── parsing_rules.yaml            # Chemical parsing rules (v1.0)
│   ├── benchmark/                    # Quick and full benchmark configurations
│   └── baselines/                    # XGBoost hyperparameter configs
│
├── data/
│   ├── raw/                          # Original CycPeptMPDB CSV files
│   ├── processed/pem_schema/         # PEM-converted JSONL + Parquet
│   └── splits/                       # Train/val/test JSONL splits
│
├── results/                          # All pre-computed outputs
│   └── benchmark/
│       ├── audit/                    # per_sample_audit.csv, excluded_samples.csv, etc.
│       ├── random/seed_*/            # 5-seed random split results per model
│       ├── sequence_cluster/seed_*/  # 5-seed cluster split results per model
│       ├── estimator_comparison/     # 5 estimators × 6 feature sets × 5 seeds
│       ├── feature_importance/       # SHAP summaries, top-20, group contributions
│       ├── time_forward_ranking/     # Year-cutoff ranking results
│       ├── scaffold_ranking/         # Per-family ranking results
│       └── summary/                  # Aggregated tables and plots
│
├── figures/
│   ├── Figure_1-4_*.{pdf,png}       # Original publication figures
│   ├── revision/                     # Current revision figures (Figs 2-6, S1-S4)
│   └── source_data/                  # CSV source data for figures
│
├── supplement/                       # Supplementary tables S1-S7 (CSV)
├── docs/                             # Technical documentation + audit reports
└── tests/                            # Unit tests
```

---

## 3. Data Availability

**All data needed to reproduce the paper's results is included in this repository.**

| Data | Location | Description |
|------|----------|-------------|
| Raw dataset | `data/raw/CycPeptMPDB_Peptide_Assay_PAMPA.csv` | Original CycPeptMPDB PAMPA subset |
| Full dataset | `data/raw/CycPeptMPDB_Peptide_All.csv` | Complete CycPeptMPDB for census |
| PEM-converted | `data/processed/pem_schema/cycpeptmpdb_pampa.{jsonl,parquet}` | Standardized schema |
| Splits | `data/splits/CycPeptMPDB_PAMPA/` | Random split: 5,056 / 1,084 / 1,084 (train/val/test). Cluster split (70% seq identity): 4,801 / 1,306 / 1,117 (train/val/test) |
| Audit outputs | `results/benchmark/audit/` | per_sample_audit.csv, excluded_samples.csv, monomer_coverage.csv, audit_summary.json |
| Model outputs | `results/benchmark/{random,sequence_cluster}/seed_*/` | Per-seed metrics, predictions, params |
| Result tables | `results/benchmark/summary/` | Aggregated tables (benchmark, ablation, model family) |
| Figures | `figures/revision/` | All main + supplementary figures (PNG) |

**No external downloads are required** to regenerate figures, tables, or statistical analyses from pre-computed results. Raw data is included for full pipeline reproduction.

**Source:** CycPeptMPDB (https://cycpeptmpdb.com) — PAMPA subset. Please consult CycPeptMPDB terms before redistribution of the raw data.

---

## 4. Environment Setup

**Python 3.8+** required.

```bash
# Clone and enter the repository
cd CASPer_re

# Install core dependencies
pip install -r requirements.txt
```

Minimal working dependencies (for smoke test and figure regeneration):
```
numpy, pandas, scipy, scikit-learn, xgboost, matplotlib, seaborn, pyyaml, rdkit, optuna, shap
```

For full reproduction (including supplementary controls and cluster split):
```
# Additional: tqdm, joblib
pip install tqdm joblib
```

---

## 5. Reproduction Workflow

All commands are run from the **project root**.

### 5.1 Smoke Test (~15 min, uses pre-computed results)

Regenerate all tables and figures without re-training models:

```bash
# Audit
python scripts/run_full_audit.py

# Quick benchmark (10 Optuna trials, 3 seeds)
python scripts/run_benchmark.py --quick

# Generate summary tables
python scripts/run_benchmark.py --summary-only

# Estimator comparison
python scripts/run_estimator_comparison.py

# SHAP analysis
python scripts/run_shap_analysis.py

# Ranking simulations
python scripts/run_scaffold_ranking.py

# Generate all figures
python scripts/make_revision_figures.py
```

### 5.2 Full Reproduction (~2-6 hours)

#### Step 1: Data preparation
```bash
python scripts/convert_to_pem_schema.py --dataset cycpeptmpdb
python scripts/02_generate_splits.py
```

#### Step 2: Full audit
```bash
python scripts/run_full_audit.py
python scripts/run_strict_census.py
```
Outputs: `results/benchmark/audit/` (Table 1 data)

#### Step 3: Main benchmarks
```bash
# Random split (all models, 50 Optuna trials, 5 seeds)
python scripts/run_benchmark.py --full

# Sequence-cluster split
python scripts/create_sequence_cluster_split.py
python scripts/run_benchmark.py --full --split sequence_cluster
```
Outputs: `results/benchmark/{random,sequence_cluster}/seed_*/` (Tables 2-3 data)

#### Step 4: Estimator comparison
```bash
python scripts/run_estimator_comparison.py
```
Outputs: `results/benchmark/estimator_comparison/comparison_summary.csv` (Table 5 data)

#### Step 5: SHAP analysis
```bash
python scripts/run_shap_analysis.py
```
Outputs: `results/benchmark/feature_importance/` (Table 4 data, Fig 4)

#### Step 6: Ranking simulations
```bash
python scripts/run_scaffold_ranking.py
```
Outputs: `results/benchmark/{time_forward_ranking,scaffold_ranking}/` (Tables 6-7 data, Fig 6)

#### Step 7: Supplementary controls
```bash
python scripts/run_all_supplement_5seed.py
python scripts/run_perturbation_seeds.py
```
Outputs: `results/supplement_5seed/`, `results/perturbation/` (Figs S2-S4)

#### Step 8: Statistical analysis
```bash
python scripts/compute_paired_significance_tests.py
python scripts/compute_bootstrap_confidence_intervals.py
```

#### Step 9: Generate figures
```bash
python scripts/make_revision_figures.py
```
Outputs: `figures/revision/{fig2-6,figS1-S4}.png`

#### Step 10: Compile final summary
```bash
python scripts/compile_all_results.py
python scripts/compile_final_summary.py
```

---

## 6. Main Outputs — Manuscript Correspondence

### Tables

| Manuscript | Output File | Script |
|------------|-------------|--------|
| Table 1 — Dataset Census | `results/benchmark/audit/audit_summary.json` | `run_full_audit.py` |
| Table 2 — Descriptor Block Summary | `src/baselines/featurizers/anchor_aware_descriptors.py` | (feature definitions) |
| Table 3 — Random-Split Benchmark | `results/benchmark/summary/benchmark_table_for_manuscript.csv` | `run_benchmark.py --summary-only` |
| Table 4 — Cluster-Split Benchmark | `results/benchmark/summary/mean_std_by_model.csv` (cluster rows) | `run_benchmark.py --summary-only` |
| Table 5 — SHAP Attribution | `results/benchmark/feature_importance/feature_attribution_table_for_manuscript.csv` | `run_shap_analysis.py` |
| Table 6 — Estimator Comparison | `results/benchmark/estimator_comparison/comparison_summary.csv` | `run_estimator_comparison.py` |
| Table 7 — Time-Forward Ranking | `results/benchmark/time_forward_ranking/summary_by_model.csv` | `run_scaffold_ranking.py` |
| Table 8 — Scaffold-Focused Ranking | `results/benchmark/scaffold_ranking/summary_by_model.csv` | `run_scaffold_ranking.py` |

### Figures

| Manuscript | Output File | Script |
|------------|-------------|--------|
| Fig 1 — Problem Framing | `figures/Figure_1_graded_perturbation.pdf` | `generate_publication_figures.py` |
| Fig 2 — Random vs Cluster Slope | `figures/revision/fig2_random_cluster_slope.png` | `make_revision_figures.py` |
| Fig 3 — Descriptor Ablation | `figures/revision/fig3_descriptor_ablation.png` | `make_revision_figures.py` |
| Fig 4 — SHAP Attribution | `figures/revision/fig4_shap_group_attribution.png` | `make_revision_figures.py` |
| Fig 5 — Estimator Heatmap | `figures/revision/fig5_estimator_heatmap.png` | `make_revision_figures.py` |
| Fig 6 — Temporal/Scaffold Ranking | `figures/revision/fig6_temporal_scaffold_ranking.png` | `make_revision_figures.py` |
| Fig S1 — Parser Audit Flow | `figures/revision/figS1_parser_audit_flow.png` | `make_revision_figures.py` |
| Fig S2 — Site Perturbation Controls | `figures/revision/figS2_site_perturbation_controls.png` | `make_revision_figures.py` |
| Fig S3 — B-Subblock Ablation | `figures/revision/figS3_B_subblock_ablation.png` | `make_revision_figures.py` |
| Fig S4 — Predicted vs Observed | `figures/revision/figS4_predicted_vs_observed.png` | `make_revision_figures.py` |

### Supplementary Tables

| Manuscript | File | Script |
|------------|------|--------|
| Table S1 — Position-Only Control | `results/supplement_5seed/tuning_Position-only_seed*/metrics.json` | `run_all_supplement_5seed.py` |
| Table S2 — Wrong-Site / Coarse-Site Control | `results/supplement_5seed/tuning_{Wrong,Coarse}*_seed*/metrics.json` | `run_all_supplement_5seed.py` |
| Table S3 — Graded Anchor-Shift Perturbation | `results/supplement_5seed/tuning_Shift*_seed*/metrics.json` | `run_all_supplement_5seed.py` |
| Table S4 — B-Subblock Ablation | `results/supplement_5seed/tuning_A+B{1,2,3}_seed*/metrics.json` | `run_all_supplement_5seed.py` |
| Table S5 — Complete Model Inventory | `supplement/Supplementary_Table_S1_feature_inventory.csv` | `make_supplementary_table_S1.py` |

---

## 7. Notes on Scope & Boundaries

### What this repository does
- Parses and audits the CycPeptMPDB-PAMPA dataset with known anchor annotations
- Corrects monomer-to-backbone mapping errors
- Constructs site-conditioned descriptors (Groups A, B, C) for peptides with resolvable modification sites
- Benchmarks chemistry-only, sequence-only, and site-conditioned models under identical splits, seeds, and HPO
- Provides mechanistic controls (position-only, wrong-site, coarse-site, graded shift, B-subblock ablation)
- Measures SHAP group-level feature attribution
- Simulates time-forward and scaffold-focused ranking scenarios

### What this repository does NOT do
- **De novo site inference** — this study assumes modification sites are *known* from the dataset annotations. Predicting sites from sequence alone is a separate problem.
- **SOTA leaderboard** — the goal is descriptor decomposition, not claiming the best possible permeability predictor.
- **Uncertain-site handling** — peptides where anchor positions cannot be resolved are excluded (74 records) with documented reasons.


---

## 8. Statistical Methods

- **Per-seed tests:** Models trained with multiple independent random seeds (5 for most models). Per-seed R² values tested with paired *t*-test or Welch's *t*-test.
- **Per-observation tests:** Wilcoxon signed-rank on per-sample squared errors from one representative run.
- **Bootstrap CIs:** 1,000 resamples, 95% confidence intervals.
- All tests two-sided, α = 0.05. See `supplement/Supplementary_Table_S3_statistical_comparisons.csv` for full details.

---

## 9. Citation

```bibtex
@article{TODO,
  title   = {Site-Conditioned Edit Chemistry for Cyclic Peptide Permeability Modeling},
  author  = {TODO},
  journal = {TODO},
  year    = {2026},
}
```

---

## 10. License

Code: [MIT License](LICENSE).
The CycPeptMPDB dataset is subject to its own terms; consult [cycpeptmpdb.com](https://cycpeptmpdb.com) before redistribution.
