# Data Splitting - Quick Reference

## Quick Start

```bash
# Generate all splits
python scripts/04_generate_splits.py

# Single dataset
python scripts/04_generate_splits.py --dataset CycPeptMPDB_PAMPA

# Generate reports
python scripts/04_split_summary_report.py
```

## Available Strategies

| Strategy | Use When | Risk | Min Requirements |
|----------|----------|------|------------------|
| `random` | Baseline only | HIGH | n ≥ 10 |
| `scaffold_aware` | Test new scaffolds | LOW | ≥20 scaffolds |
| `sequence_cluster` | Test dissimilar seqs | LOW | ≥20 clusters |
| `edit_family_aware` | Balanced edits | MED | ≥10/profile |
| `same_edit_family_new_scaffold` | Transferable edits | LOW | ≥5 scaffolds/family |
| `grouped_derivative` | Derivative series | VERY LOW | ≥10% in groups |

## Python API

```python
from src.data.serialization import load_samples_jsonl
from src.data.splitting import ScaffoldAwareSplitter

# Load data
samples = load_samples_jsonl('data/processed/chem_repr/dataset.jsonl')

# Create splitter
splitter = ScaffoldAwareSplitter(min_scaffolds=20, random_seed=42)

# Check feasibility
is_feasible, reason = splitter.check_feasibility(samples)

# Generate split
if is_feasible:
    result = splitter.split(samples)

    # Access splits
    train = result.train_samples
    val = result.val_samples
    test = result.test_samples

    # Check leakage
    leakage = result.metadata.leakage_analysis
    print(f"Risk: {leakage.overall_risk}")
```

## Leakage Analysis

**Risk Levels**:
- **LOW**: Max identity <70%, scaffold overlap <10%
- **MODERATE**: Max identity 70-85%, scaffold overlap 10-50%
- **HIGH**: Max identity >85% OR scaffold overlap >50%

**Metrics**:
- Sequence similarity (max, mean, % high)
- Scaffold overlap (shared count, % test)
- Edit pattern similarity (cosine, JS divergence)
- Label distribution (KS test)

## Output Structure

```
data/splits/{dataset}/{strategy}/
├── train.jsonl           # Training samples
├── val.jsonl            # Validation samples
├── test.jsonl           # Test samples
└── metadata.json        # Complete metadata

data/splits/{dataset}/
└── split_metadata.json  # Summary across strategies

reports/splits/
└── {dataset}_split_summary.md  # Markdown report
```

## Common Issues

### "Only X scaffolds (minimum 20 required)"
→ Use `sequence_cluster` or `random` instead

### "Clustering failed"
→ Install MMseqs2: `conda install -c bioconda mmseqs2`

### "Proportions too far from target"
→ Increase sample size or adjust min requirements

### Pydantic version error
→ Downgrade: `pip install 'pydantic<2'`
→ Or update schema to pydantic v2

## Dataset Recommendations

**CycPeptMPDB-PAMPA**:
1. Try `scaffold_aware` first
2. Fall back to `edit_family_aware`
3. Use `random` as baseline

**PepMSND**:
1. Try `sequence_cluster` first
2. Fall back to `edit_family_aware`
3. Use `random` as baseline

**DBAASP**:
1. Try `grouped_derivative` first
2. Fall back to `sequence_cluster`
3. Be prepared for infeasibility

## Full Documentation

- **Protocol**: `docs/split_protocol.md`
- **Implementation**: `STAGE4_SPLITS.md`
- **API**: Docstrings in `src/data/splitting/`
- **Tests**: `tests/test_splitting.py`
