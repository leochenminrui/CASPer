# PEM Data Splitting Protocol

**Version**: 1.0.0
**Date**: 2026-04-02
**Stage**: 4 - Realistic Data Splits

## Philosophy

### Beyond Random Splits

**Problem**: Random splits can leak information through:
- Near-duplicate sequences
- Same-scaffold variants with different edits
- Sequence clustering at high identity

**Solution**: Multiple split strategies with explicit leakage analysis

### Conservative Reporting

- If a realistic split is infeasible, explicitly report why
- Document all clustering thresholds
- Quantify leakage risk for each strategy
- Prefer honest "not feasible" over questionable splits

## Split Strategies

### 1. Random Split (Baseline)

**Method**: Stratified random sampling

**Parameters**:
- Train: 70%
- Val: 15%
- Test: 15%
- Stratification: By label quartiles
- Random seed: 42

**When to use**:
- Baseline comparison
- Small datasets where other strategies infeasible
- When data is truly independent

**Leakage risk**: High if sequences are similar

**Implementation**:
```python
sklearn.model_selection.train_test_split(
    stratify=label_quartiles,
    random_state=42
)
```

### 2. Sequence Cluster Split

**Method**: Cluster sequences, split by cluster

**Parameters**:
- Clustering method: CD-HIT or MMseqs2
- Identity threshold: 0.7 (default, adjustable)
- Assignment: All samples in same cluster → same split

**Algorithm**:
1. Cluster sequences at threshold (e.g., 70% identity)
2. Assign clusters to train/val/test
3. All sequences in cluster go to same split

**When to use**:
- Evaluate generalization to dissimilar sequences
- Peptide datasets with sequence variation

**Leakage risk**: Low for sequence similarity

**Feasibility check**:
- Minimum cluster count: ≥20 (for reasonable split)
- If <20 clusters: Report as infeasible

### 3. Scaffold-Aware Split

**Method**: Extract backbone scaffold, split by scaffold

**Scaffold definition** (for peptides):
- Canonical sequence (no modifications)
- Cyclization topology (if applicable)

**Algorithm**:
1. Extract scaffold for each sample
2. Group by scaffold
3. Assign scaffolds to train/val/test
4. All samples with same scaffold → same split

**When to use**:
- CycPeptMPDB (cyclic peptides with modifications)
- Test generalization to new scaffolds

**Leakage risk**: Low for scaffold similarity

**Feasibility check**:
- Minimum scaffold count: ≥20
- Scaffold size distribution: Not dominated by one scaffold

### 4. Edit-Family-Aware Split

**Method**: Stratify by edit family distribution

**Algorithm**:
1. Characterize each sample by edit families present
2. Create edit profile (e.g., "sidechain+cyclization")
3. Stratify split to maintain edit profile distribution

**When to use**:
- Ensure all edit types represented in each split
- Balanced evaluation across modification types

**Leakage risk**: Moderate (same family, different samples)

**Feasibility check**:
- Each edit profile has ≥10 samples
- Edit profiles not too sparse

### 5. Same-Edit-Family / New-Scaffold Split

**Method**: Train on edit families with certain scaffolds, test on same families but new scaffolds

**Algorithm**:
1. Group by (edit_family, scaffold)
2. For each edit family:
   - Split scaffolds into train/test
   - Val samples from both train and test scaffolds
3. Ensures test has same edit types but new scaffolds

**When to use**:
- CycPeptMPDB with multiple scaffolds per edit type
- Test if model learns edit effects transferably

**Leakage risk**: Low (different scaffolds)

**Feasibility check**:
- Each edit family has ≥5 different scaffolds
- Sufficient samples per (edit_family, scaffold) pair

### 6. Grouped Derivative Split (DBAASP)

**Method**: Group same-scaffold derivatives, keep groups intact

**Algorithm**:
1. Identify derivative groups (edit distance ≤ 1)
2. Keep entire group in same split
3. Split at group level, not sample level

**When to use**:
- DBAASP if same-scaffold derivatives exist
- Prevent train/test leakage through minimal edits

**Leakage risk**: Very low (groups intact)

**Feasibility check**:
- Derivative groups identifiable (≥10% of data)
- Groups not too large (max 20% of data per group)

## Leakage Risk Analysis

### Metrics

For each split, compute:

1. **Sequence Similarity**:
   - Max pairwise identity between train and test
   - Percentage of test with >70% identity to train

2. **Scaffold Overlap**:
   - Number of shared scaffolds between splits
   - Percentage of test samples with train scaffold

3. **Edit Pattern Similarity**:
   - Cosine similarity of edit profile distributions
   - Jensen-Shannon divergence of edit families

4. **Label Distribution**:
   - KS test statistic between splits
   - Mean/std of labels in each split

### Risk Levels

- **Low risk**: Max sequence identity <70%, no scaffold overlap
- **Moderate risk**: Max identity 70-85%, minimal scaffold overlap
- **High risk**: Max identity >85%, significant scaffold overlap

## Dataset-Specific Protocols

### CycPeptMPDB (PAMPA)

**Priority splits** (in order):
1. **Scaffold-aware**: If ≥20 unique scaffolds
2. **Same-edit-family / new-scaffold**: If edit families well-distributed
3. **Edit-family-aware**: If edit profiles diverse
4. **Sequence cluster**: If ≥20 clusters at 70% identity
5. **Random**: As baseline

**Feasibility determination**:
```python
n_scaffolds = count_unique_scaffolds(samples)
if n_scaffolds >= 20:
    implement_scaffold_split()
elif n_editable_scaffolds >= 5 per edit_family:
    implement_edit_scaffold_split()
else:
    report_infeasible("Insufficient scaffold diversity")
```

**Expected outcome**:
- Likely feasible: Scaffold-aware, edit-family-aware
- Possibly feasible: Same-edit-family / new-scaffold
- Always feasible: Random

### PepMSND (Blood Stability)

**Priority splits**:
1. **Sequence cluster**: If ≥20 clusters at 70% identity
2. **Edit-aware**: If edit annotations rich
3. **Random**: As baseline

**Feasibility determination**:
```python
clusters = cluster_sequences(samples, threshold=0.7)
if len(clusters) >= 20:
    implement_cluster_split()
elif edit_annotations_rich:
    implement_edit_aware_split()
else:
    implement_random_split()
```

**Expected outcome**:
- Likely feasible: Sequence cluster, random
- Possibly feasible: Edit-aware (depends on annotations)

### DBAASP (Exploratory)

**Priority splits**:
1. **Grouped derivative**: If derivative groups identifiable
2. **Sequence cluster**: If ≥20 clusters
3. **Random or explicit infeasibility**

**Feasibility determination**:
```python
derivative_groups = identify_derivative_groups(samples)
coverage = len(in_groups) / len(samples)

if coverage >= 0.1 and max_group_size <= 0.2 * len(samples):
    implement_grouped_derivative_split()
elif can_cluster_sequences():
    implement_cluster_split()
else:
    report_explicit_infeasibility()
```

**Expected outcome**:
- Unknown: Depends on derivative structure
- Must honestly report if realistic split infeasible

## Split File Format

### Directory Structure

```
data/splits/{dataset}/
├── random/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── scaffold_aware/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── scaffold_assignments.json
├── sequence_cluster/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── cluster_assignments.json
└── split_metadata.json
```

### JSONL Format

Each split file contains PEMSample objects with added `split_metadata`:

```json
{
  "sample_id": "CYCPEPT_000001",
  "dataset": "CycPeptMPDB_PAMPA",
  "sequence": "ACDEFG...",
  "split_metadata": {
    "split_strategy": "scaffold_aware",
    "split_name": "train",
    "scaffold_id": "scaffold_042",
    "cluster_id": null,
    "random_seed": 42,
    "split_version": "1.0.0"
  },
  ...
}
```

### Metadata File

`split_metadata.json`:

```json
{
  "dataset": "CycPeptMPDB_PAMPA",
  "split_strategies": {
    "random": {
      "feasible": true,
      "train_count": 700,
      "val_count": 150,
      "test_count": 150,
      "random_seed": 42,
      "leakage_risk": "high"
    },
    "scaffold_aware": {
      "feasible": true,
      "train_count": 680,
      "val_count": 160,
      "test_count": 160,
      "n_scaffolds": 45,
      "scaffold_distribution": {...},
      "leakage_risk": "low"
    },
    "sequence_cluster": {
      "feasible": false,
      "reason": "Only 12 clusters at 70% identity threshold"
    }
  },
  "recommended_strategy": "scaffold_aware",
  "creation_date": "2026-04-02T14:30:00"
}
```

## Implementation Requirements

### 1. Reproducibility

- **Fixed random seeds**: 42 for all random operations
- **Deterministic clustering**: Same input → same clusters
- **Versioned algorithms**: Document algorithm version

### 2. Validation

For each split, verify:
- No sample appears in multiple splits
- All samples assigned to exactly one split
- Train/val/test proportions within ±2% of target
- Label distributions not significantly different (KS test p>0.05)

### 3. Documentation

For each strategy:
- Algorithm description
- Parameter values
- Feasibility assessment
- Leakage risk quantification
- Example assignments

## Leakage Prevention Checklist

Before accepting a split:

- [ ] Computed max sequence similarity train→test
- [ ] Checked for scaffold overlap
- [ ] Verified no duplicate samples across splits
- [ ] Assessed edit pattern similarity
- [ ] Quantified label distribution differences
- [ ] Documented all thresholds and parameters
- [ ] Explicit risk classification (low/moderate/high)

## Infeasibility Reporting

If a split strategy is infeasible, report must include:

1. **Strategy name**
2. **Reason for infeasibility**
3. **Quantitative evidence** (e.g., "only 8 clusters")
4. **Threshold not met** (e.g., "minimum 20 required")
5. **Alternative strategy** (if any)

Example:
```
Strategy: scaffold_aware
Status: INFEASIBLE
Reason: Insufficient scaffold diversity
Evidence: Only 12 unique scaffolds identified
Threshold: Minimum 20 required for reliable split
Alternative: Use sequence_cluster split instead
```

## Success Criteria

### Minimum Viable:
- Random split implemented for all datasets
- At least 1 advanced strategy per dataset (if feasible)
- Complete leakage analysis for all splits
- Honest infeasibility reporting where applicable

### Ideal:
- 2-3 strategies per dataset
- All strategies have low/moderate leakage risk
- Clear recommendation for each dataset
- Derivative grouping working for DBAASP

## Version History

**v1.0.0** (2026-04-02)
- Initial split protocol
- 6 split strategies defined
- Dataset-specific protocols
- Leakage risk framework
- Infeasibility reporting guidelines

## References

- Stage 2 Schema: `docs/schema_spec.md`
- Stage 3 Chem Repr: `docs/chem_repr_spec.md`
- Splitting Code: `src/data/splitting/`
