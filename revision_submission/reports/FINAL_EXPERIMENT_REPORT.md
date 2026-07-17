# Final experimental report

Experimental analyses are complete; manuscript production is blocked only by absent source files.

## Key paired R² effects

- random: Chemistry + Site + Context minus Chemistry: ΔR²=0.0412, 95% CI [0.0336, 0.0488] — supported directional difference.
- random: Chemistry + Site minus Chemistry: ΔR²=0.0207, 95% CI [-0.0103, 0.0516] — inconclusive.
- random: Chemistry + Site + Context minus Chemistry + Site: ΔR²=0.0205, 95% CI [-0.0061, 0.0472] — inconclusive.
- random: Chemistry + Context minus Chemistry: ΔR²=-0.0112, 95% CI [-0.0364, 0.0139] — inconclusive.
- random: Site + Context minus Site: ΔR²=0.0866, 95% CI [0.0416, 0.1316] — supported directional difference.
- random: Site + Context minus Context: ΔR²=0.1091, 95% CI [0.0812, 0.1369] — supported directional difference.
- sequence_cluster: Chemistry + Site minus Chemistry: ΔR²=0.0294, 95% CI [0.0100, 0.0488] — supported directional difference.
- sequence_cluster: Chemistry + Site minus Chemistry + Site + Context: ΔR²=0.0215, 95% CI [-0.0262, 0.0692] — inconclusive.
- sequence_cluster: Chemistry + Site + Context minus Chemistry: ΔR²=0.0079, 95% CI [-0.0332, 0.0489] — inconclusive.
- sequence_cluster: Chemistry + Context minus Chemistry: ΔR²=-0.0242, 95% CI [-0.1108, 0.0624] — inconclusive.
- sequence_cluster: Site + Context minus Site: ΔR²=-0.0396, 95% CI [-0.1083, 0.0290] — inconclusive.
- sequence_cluster: Site + Context minus Context: ΔR²=0.0177, 95% CI [-0.0667, 0.1022] — inconclusive.
- sequence_cluster: Chemistry + Site minus ECFP: ΔR²=0.0833, 95% CI [0.0513, 0.1154] — supported directional difference.
- sequence_cluster: Chemistry + Site minus RDKit: ΔR²=0.1312, 95% CI [0.0431, 0.2193] — supported directional difference.
- random_10_trial: Random Forest minus XGBoost for Chemistry + Site + Context: ΔR²=0.0014, 95% CI [-0.0092, 0.0120] — inconclusive.

## Estimator robustness

Random Forest and XGBoost were indistinguishable for the complete representation: the paired R² interval included zero. Tree/kernel nonlinear estimators had substantially higher point estimates than Ridge and ElasticNet.

## SHAP

For the complete model, Chemistry was the largest individual subblock, while combined Site was the largest conceptual group. SHAP values are fitted-model attribution and are not causal.

## Scaffold ranking

Chemistry + Site minus Chemistry: -0.0170, family-bootstrap 95% CI [-0.0432, 0.0087]. The interval included zero.

## Time-forward

ECFP had the highest mean cutoff-level Spearman, but performance collapsed at the 2022 and 2023 cutoffs. The 2021 cutoff coincided with a sharp decrease in median nearest-neighbor Tanimoto similarity; 2022–2023 also had much smaller test sets and shifted label distributions. These diagnostics do not identify a unique cause.
