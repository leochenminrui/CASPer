# Minor Revision Experimental Report

## A. Executive status

PARTIALLY COMPLETE

The audit verified the corrected 7,224-sample dataset and canonical seeds 0–4. 35 of 70 primary-ablation seed cells and 125 of 175 estimator-matrix cells are protocol-matched and reusable. Missing full-protocol training was not replaced by lower-budget results.

## B. Reused versus newly run experiments

Reused 35 primary cells: committed 50-trial random/cluster A, A+B, and A+B+C cells, plus random A+C supplementary cells with five seeds and 50 trials. Reused estimator cells are listed row-by-row in `estimator_matrix/seed_level_results.csv`; A+B and A+C were absent from the committed 10-trial matrix. No performance run was newly completed.

## C. Complete primary-ablation results

The requested seven-way table is incomplete. See `primary_ablation/summary_with_ci.csv`. 35 cells are BLOCKED.

## D. Key paired comparisons

Available five-seed paired results and 95% t intervals are in `paired_statistics/paired_differences.csv`. Comparisons lacking both protocol-matched sides are omitted and remain inconclusive.

## E. Estimator conclusions

Random Forest versus XGBoost for A+B+C is available with paired seed analysis possible from the machine-readable table; rounded means alone are not treated as reliable superiority. The incomplete A+B/A+C matrix prevents a complete estimator-dependence conclusion.

## F. Scaffold-ranking conclusion

See `scaffold_ranking/paired_comparison_AB_vs_A.csv`; inference uses peptide-family bootstrap and a sign test.

## G. Time-forward diagnosis

BLOCKED for revision-grade inference: committed cutoff summaries contain eight cutoffs but no compound-level predictions, and the existing generator is non-runnable (wrong raw path and undefined variables). CIs and jointly grouped model differences cannot be reconstructed without rerunning.

## H. SHAP conclusion

BLOCKED for five-seed uncertainty: committed SHAP tables are single aggregate analyses. Existing full-model attribution reports A=0.3287, C=0.2498, B1=0.1751, B2=0.1417, B3=0.1047; combined B=0.4215, so B—not A—is the largest conceptual group, while A is the largest individual subblock. SHAP is model attribution, not causal evidence.

## I. Group A versus RDKit

The code inventory shows both operate on full-peptide SMILES. Group A is a selected 8-descriptor whole-molecule panel plus two edit-count features; RDKit is the broad dynamically discovered 2D descriptor set with training-median imputation. Registry claims of per-edit aggregation do not match implementation.

## J. Manuscript-safe statements

- Claim: Combined Group B is larger than Group A in the committed full-model SHAP summary. Status: SUPPORTED WITH CAUTION. Evidence: B1+B2+B3=0.4215 versus A=0.3287; single aggregate analysis, no five-seed CI.
- Claim: Chemistry is the largest overall conceptual SHAP group. Status: NOT SUPPORTED.
- Claim: The seven-way ablation is complete. Status: NOT SUPPORTED.
- Claim: The estimator-by-descriptor matrix is complete. Status: NOT SUPPORTED.

## K. Remaining blockers

- Primary ablation: 35 missing seed cells; measured first full-search trial took about 31 seconds, projecting roughly 17 CPU-hours for the missing 2,000 trials.
- Estimator matrix: 50 missing A+B/A+C cells.
- Time-forward: no compound predictions and broken legacy generator.
- SHAP: no five-seed SHAP artifacts/models; retraining required.
- Existing prediction files lack explicit sequence-cluster identifiers; cluster assignments are not shipped separately.
