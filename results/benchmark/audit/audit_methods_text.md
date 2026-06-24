# Dataset Audit — Methods

## Automated Audit Protocol
All 7,298 input rows were processed programmatically without manual sampling.

1. Monomer Census: 277 unique monomers, 173 mapped (2 unmapped after v2.0)
2. Per-Sample Parsing: 7,224 success (99.0%), 74 failed (1.0%)
3. Anchor Classification: 4,920/7,224 explicit anchors (68.1%)
4. Quality Flags: {'duplicate_sequence': 108}

## Exclusion Transparency
All 74 excluded rows with reasons in `excluded_samples.csv`.
No rows were silently dropped.
