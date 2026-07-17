# Feature Attribution — Response to Reviewer 1

**Reviewer Comment:** "Feature importance already exists in code but not analyzed in manuscript."

**Response:** We have added systematic feature attribution analysis for all key models.

## Methods
- Method: SHAP TreeExplainer
- SHAP available: True

## Key Findings
- **chem_A_xgb**: dominant group = A_Chem (100.0% of total attribution)
- **chem_site_AB_xgb**: dominant group = A_Chem (42.8% of total attribution)
- **full_ABC_xgb**: dominant group = A_Chem (32.9% of total attribution)
- **ecfp_xgb**: dominant group = Other (95.6% of total attribution)
- **rdkit_full_xgb**: dominant group = B2_ResidueComp (60.8% of total attribution)

## Interpretation Boundaries
- Feature attribution supports but does not prove the chemistry-dominant interpretation.
- Attribution is model-dependent and should not be interpreted as causal evidence.
- SHAP values reflect the model's learned dependence structure, not ground-truth causal mechanisms.
- For the full_ABC_xgb model, Group B+C features together account for meaningful attribution,
  consistent with the ablation study showing site and context contributions.

## Output Files
- `results/benchmark/feature_importance/shap_summary_*.pdf` — SHAP summary plots
- `results/benchmark/feature_importance/shap_top20_*.csv` — Top-20 features per model
- `results/benchmark/feature_importance/shap_group_contribution_*.csv` — Group-level contributions
- `results/benchmark/feature_importance/feature_attribution_table_for_manuscript.csv` — Manuscript table