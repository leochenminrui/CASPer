# Manuscript-to-Code Mapping

**Paper:** Site-Conditioned Edit Chemistry for Cyclic Peptide Permeability Modeling
**Repository:** CASPer
**Table/Figure numbering:** matches final manuscript

This document maps every manuscript table, figure, and claim to the responsible scripts, inputs, and outputs.

---

## Main Tables

### Table 1 — Dataset Census

7,298 raw → 7,224 site-resolved after monomer-to-backbone correction, 74 excluded, 1,325 unique sequences.

| Item | Script | Input | Output |
|------|--------|-------|--------|
| Full per-sample audit (all rows) | `scripts/run_full_audit.py` | `data/raw/CycPeptMPDB_Peptide_Assay_PAMPA.csv` | `results/benchmark/audit/per_sample_audit.csv` |
| Excluded records with reasons | same | same | `results/benchmark/audit/excluded_samples.csv` |
| Monomer-to-backbone coverage | same | same | `results/benchmark/audit/monomer_coverage.csv` |
| Aggregate statistics | same | same | `results/benchmark/audit/audit_summary.json`, `audit_summary.csv` |
| Strict anchor-resolvability breakdow | `scripts/run_strict_census.py` | same | Supplementary census report |

**Verification:**
```bash
python scripts/run_full_audit.py
```

---

### Table 2 — Descriptor Block Summary

Group definitions, dimensionality, and example features for Blocks A / B1 / B2 / B3 / C.

| Item | Source |
|------|--------|
| Feature definitions | `src/baselines/featurizers/anchor_aware_descriptors.py` |
| Supplementary detail | `docs/anchor_aware_descriptor_schema.md`, `supplement/Supplementary_Table_S1_feature_inventory.csv` |

---

### Table 3 — Random-Split Benchmark

Random 70/15/15 split (5,056 train / 1,084 val / 1,084 test), mean ± SD across 5 seeds, Optuna HPO for all models.

| Item | Script | Input | Output |
|------|--------|-------|--------|
| Train all models (unified) | `scripts/run_benchmark.py --full` | `data/splits/CycPeptMPDB_PAMPA/random/` | `results/benchmark/random/seed_*/` |
| Summary table | `scripts/run_benchmark.py --summary-only` | Above | `results/benchmark/summary/benchmark_table_for_manuscript.csv` |
| Mean ± SD per model | same | same | `results/benchmark/summary/mean_std_by_model.csv` |

**Models in this table:**
`seq_aa_xgb`, `position_only_xgb`, `chem_A_xgb`, `site_B_xgb`, `context_C_xgb`, `site_context_BC_xgb`, `chem_site_AB_xgb`, `chem_context_AC_xgb`, `full_ABC_xgb`, `chem_B1_xgb`, `chem_B2_xgb`, `chem_B3_xgb`, `ecfp_xgb`, `rdkit_full_xgb`, `ecfp_rf`

**Verification:**
```bash
python scripts/run_benchmark.py --full               # full training
python scripts/run_benchmark.py --quick               # smoke test
python scripts/run_benchmark.py --summary-only         # regenerate table from existing results
```

---

### Table 4 — Cluster-Split Benchmark

70% sequence-identity cluster split (4,801 train / 1,306 val / 1,117 test).

| Item | Script | Input | Output |
|------|--------|-------|--------|
| Generate cluster split | `scripts/create_sequence_cluster_split.py` | PEM samples | `data/splits/CycPeptMPDB_PAMPA/sequence_cluster/` |
| Benchmark on cluster split | `scripts/run_benchmark.py --full --split sequence_cluster` | Cluster split | `results/benchmark/sequence_cluster/seed_*/` |
| Cluster vs random comparison | `scripts/eval_harder_split.py` | Both split results | Comparison metrics |

**Verification:**
```bash
python scripts/create_sequence_cluster_split.py
python scripts/run_benchmark.py --full --split sequence_cluster
```

---

### Table 5 — SHAP Group-Level Attribution

| Item | Script | Input | Output |
|------|--------|-------|--------|
| SHAP TreeExplainer analysis | `scripts/run_shap_analysis.py` | Trained XGBoost models | `results/benchmark/feature_importance/` |
| Group contributions | same | same | `shap_group_contribution_full_ABC_xgb.csv`, `shap_group_contribution_chem_site_AB_xgb.csv` |
| Top-20 feature importance | same | same | `shap_top20_*.csv` |
| Manuscript-ready table | same | same | `feature_attribution_table_for_manuscript.csv` |

**Verification:**
```bash
python scripts/run_shap_analysis.py
```

---

### Table 6 — Estimator Comparison

Five estimator families (Ridge, ElasticNet, Random Forest, RBF-SVR, XGBoost) × 6 feature sets (AA Comp, Chem, Site, Context, Site+Context, Chem+Site+Context). All with Optuna HPO, 10 trials, 5 seeds.

| Item | Script | Input | Output |
|------|--------|-------|--------|
| Estimator comparison | `scripts/run_estimator_comparison.py` | `data/splits/CycPeptMPDB_PAMPA/random/` | `results/benchmark/estimator_comparison/comparison_summary.csv` |

**Verification:**
```bash
python scripts/run_estimator_comparison.py
```

---

### Table 7 — Time-Forward Ranking

Train on earlier publication years, test on later years (8 cutoff years).

| Item | Script | Input | Output |
|------|--------|-------|--------|
| Time-forward simulation | `scripts/run_scaffold_ranking.py` | Raw CSV + PEM schema | `results/benchmark/time_forward_ranking/cutoff_level_results.csv` |
| Model summary | same | same | `results/benchmark/time_forward_ranking/summary_by_model.csv` |

**Verification:**
```bash
python scripts/run_scaffold_ranking.py
```

---

### Table 8 — Scaffold-Focused Ranking

Within-campaign ranking on novel scaffolds (49 families, min 8 members per family, historical support set + support set → test set).

| Item | Script | Input | Output |
|------|--------|-------|--------|
| Per-family ranking | `scripts/run_scaffold_ranking.py` | Same as Table 7 | `results/benchmark/scaffold_ranking/family_level_results.csv` |
| Skipped families with reasons | same | same | `results/benchmark/scaffold_ranking/skipped_scaffolds_with_reason.csv` |
| Model summary | same | same | `results/benchmark/scaffold_ranking/summary_by_model.csv` |

---

## Main Figures

| Figure | Generated by | Input data | Output file |
|--------|-------------|------------|-------------|
| Fig 1 — Problem framing | `scripts/generate_publication_figures.py` | `figures/source_data/Figure_1_source_data.csv` | `figures/Figure_1_graded_perturbation.pdf` |
| Fig 2 — Random vs cluster slope | `scripts/make_revision_figures.py` | `results/benchmark/summary/mean_std_by_model.csv` | `figures/revision/fig2_random_cluster_slope.png` |
| Fig 3 — Descriptor ablation | `scripts/make_revision_figures.py` | `mean_std_by_model.csv` + `comparison_summary.csv` | `figures/revision/fig3_descriptor_ablation.png` |
| Fig 4 — SHAP group attribution | `scripts/make_revision_figures.py` | `shap_group_contribution_*.csv` | `figures/revision/fig4_shap_group_attribution.png` |
| Fig 5 — Estimator heatmap | `scripts/make_revision_figures.py` | `comparison_summary.csv` | `figures/revision/fig5_estimator_heatmap.png` |
| Fig 6 — Temporal & scaffold ranking | `scripts/make_revision_figures.py` | `cutoff_level_results.csv` + `family_level_results.csv` | `figures/revision/fig6_temporal_scaffold_ranking.png` |

---

## Supplementary Tables

| Table | Content | Source |
|-------|---------|--------|
| **Table S1** — Position-only control | Tests whether site-aware gain is a positional shortcut (6-dim anchor count + site stats only) | `scripts/run_all_supplement_5seed.py` |
| **Table S2** — Wrong-site / coarse-site control | Binary site-perturbation: correct vs wrong vs coarse anchor positions, chemistry held fixed | `scripts/run_all_supplement_5seed.py` |
| **Table S3** — Graded anchor-shift perturbation | Systematic shift {-5, -3, -2, -1, 0, +1, +2, +3, +5} residues, R² vs |shift| | `scripts/run_all_supplement_5seed.py` |
| **Table S4** — B-subblock ablation | B1 (position), B2 (residue composition), B3 (residue properties) individual contributions | `scripts/run_all_supplement_5seed.py` |
| **Table S5** — Complete model inventory | All model variants, descriptor sets, hyperparameters, and seed counts | `supplement/Supplementary_Table_S1_feature_inventory.csv`, `scripts/make_supplementary_table_S1.py` |

---

## Supplementary Figures

| Figure | Generated by | Input data | Output file |
|--------|-------------|------------|-------------|
| Fig S1 — Parser audit flow | `scripts/make_revision_figures.py` | `results/benchmark/audit/audit_summary.json` | `figures/revision/figS1_parser_audit_flow.png` |
| Fig S2 — Site perturbation controls | `scripts/make_revision_figures.py` | `supplement/Supplementary_Table_S7_local_controls.csv` | `figures/revision/figS2_site_perturbation_controls.png` |
| Fig S3 — B-subblock ablation | `scripts/make_revision_figures.py` | `supplement/Supplementary_Table_S7_local_controls.csv` | `figures/revision/figS3_B_subblock_ablation.png` |
| Fig S4 — Predicted vs observed | `scripts/make_revision_figures.py` | `results/benchmark/*/seed_*/predictions.csv` | `figures/revision/figS4_predicted_vs_observed.png` |

---

## Supplementary Mechanistic Controls

All controls: `scripts/run_all_supplement_5seed.py` + `src/data/mechanism_controls.py`.

| Control | Feature set | Purpose |
|---------|------------|---------|
| Position-only | Anchor count + site-location stats (6 dim) | Tests positional shortcut |
| Wrong-site | Random anchor perturbation | Negative control |
| Coarse-site | Region indicator instead of exact position | Specificity control |
| Graded anchor-shift | Systematic shift {-5, …, +5} | Dose-response of anchor precision |
| B-subblock ablation | B1 / B2 / B3 individually | Subcomponent contribution |

---

## Descriptor Groups

| Group | Dim | Content | Source |
|-------|-----|---------|--------|
| A — Edit Chemistry | 10 | MW, logP, TPSA, edit counts, etc. | `src/baselines/featurizers/anchor_aware_descriptors.py` |
| B1 — Position Statistics | 6 | Anchor count, density, mean, std, range | same |
| B2 — Residue Composition | 20 | Per-AA frequencies at anchor positions | same |
| B3 — Residue Properties | 9 | Hydrophobicity, charge, polarity, aromaticity at anchors | same |
| C — Scaffold/Multi-Edit Context | 28 | Edit family distribution, type entropy, cyclization | same |
| **A+B+C — Full** | **73** | Complete site-conditioned descriptor | same |

---

## Complete Reproduction Workflow

### Smoke test (~15 min, uses pre-computed results)
```bash
python scripts/run_full_audit.py
python scripts/run_benchmark.py --quick
python scripts/run_benchmark.py --summary-only
python scripts/run_estimator_comparison.py
python scripts/run_shap_analysis.py
python scripts/run_scaffold_ranking.py
python scripts/make_revision_figures.py
```

### Full reproduction (~2-6 hours)
```bash
# 1. Data preparation
python scripts/convert_to_pem_schema.py --dataset cycpeptmpdb
python scripts/02_generate_splits.py

# 2. Audit (→ Table 1)
python scripts/run_full_audit.py

# 3. Main benchmarks (→ Tables 3-4)
python scripts/run_benchmark.py --full
python scripts/run_benchmark.py --full --split sequence_cluster

# 4. Estimator comparison (→ Table 6)
python scripts/run_estimator_comparison.py

# 5. SHAP analysis (→ Table 5)
python scripts/run_shap_analysis.py

# 6. Ranking simulations (→ Tables 7-8)
python scripts/run_scaffold_ranking.py

# 7. Supplementary controls (→ Tables S1-S4)
python scripts/run_all_supplement_5seed.py

# 8. Statistical tests
python scripts/compute_bootstrap_confidence_intervals.py
python scripts/compute_paired_significance_tests.py

# 9. All figures (→ Figs 2-6, S1-S4)
python scripts/make_revision_figures.py
```

---

## Notes

1. **All data is included** — no external downloads needed to reproduce results.
2. **Cluster split counts** match manuscript Table 1: 4,801 / 1,306 / 1,117. Random split: 5,056 / 1,084 / 1,084.
3. **Active model registry** contains 15 models across 4 roles (internal_baseline, primary_paper, ablation_control, generic_benchmark).
