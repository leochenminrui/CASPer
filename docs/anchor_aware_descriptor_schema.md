# Anchor-Aware Descriptor Feature Schema

**Version:** 1.0.0
**Date:** April 4, 2026
**Dataset:** CycPeptMPDB PAMPA
**Purpose:** Define feature extraction for anchor-aware descriptor + XGBoost model

---

## Overview

This schema defines a comprehensive feature set that combines:
1. **Global chemistry features** from whole-molecule descriptors
2. **Anchor-aware local features** from edit positions and contexts
3. **Attachment-aware multi-edit features** from edit structure

The goal is to create a principled, interpretable feature set that explicitly incorporates anchor and attachment information while maintaining the strong chemical descriptor signal from the baseline.

---

## Feature Groups

### A. Global Chemistry Features (10 features)

These features capture whole-molecule chemical properties and are computed from the SMILES representation.

| Feature Name | Source Field | Computation | Description |
|-------------|-------------|-------------|-------------|
| `mol_weight` | `assay_metadata.smiles` | RDKit Descriptors.MolWt | Molecular weight |
| `logp` | `assay_metadata.smiles` | RDKit Descriptors.MolLogP | Partition coefficient (lipophilicity) |
| `tpsa` | `assay_metadata.smiles` | RDKit Descriptors.TPSA | Topological polar surface area |
| `num_h_acceptors` | `assay_metadata.smiles` | RDKit Descriptors.NumHAcceptors | Number of H-bond acceptors |
| `num_h_donors` | `assay_metadata.smiles` | RDKit Descriptors.NumHDonors | Number of H-bond donors |
| `num_rotatable_bonds` | `assay_metadata.smiles` | RDKit Descriptors.NumRotatableBonds | Number of rotatable bonds |
| `num_aromatic_rings` | `assay_metadata.smiles` | RDKit rdMolDescriptors.CalcNumAromaticRings | Number of aromatic rings |
| `num_aliphatic_rings` | `assay_metadata.smiles` | RDKit rdMolDescriptors.CalcNumAliphaticRings | Number of aliphatic rings |
| `num_edits` | `len(sample.edits)` | Count | Total number of edits |
| `num_edit_families` | `set(e.edit_family for e in sample.edits)` | Count of unique families | Edit family diversity |

**Rationale:** These features represent the strong baseline signal from descriptor-only XGBoost.

---

### B. Anchor-Aware Local Features (35 features)

These features explicitly encode anchor position and residue information.

#### B1. Anchor Position Statistics (6 features)

| Feature Name | Source Field | Computation | Description |
|-------------|-------------|-------------|-------------|
| `anchor_count_total` | `edits[*].anchor_positions` | Count of all anchor positions | Total anchors across all edits |
| `anchor_count_unique` | `edits[*].anchor_positions` | Count of unique positions | Number of unique anchor sites |
| `anchor_density` | `anchor_count_unique / len(sequence)` | Ratio | Fraction of sequence with edits |
| `anchor_pos_mean` | `edits[*].anchor_positions` | Mean | Average anchor position |
| `anchor_pos_std` | `edits[*].anchor_positions` | Std dev | Spread of anchor positions |
| `anchor_pos_range` | `edits[*].anchor_positions` | Max - Min | Position span |

#### B2. Anchor Residue Composition (20 features)

One-hot encoding for each standard amino acid at anchor positions.

| Feature Name | Source Field | Computation | Description |
|-------------|-------------|-------------|-------------|
| `anchor_res_A` through `anchor_res_Y` | `edits[*].anchor_residues` | Count / total anchors | Frequency of each AA at anchors |

**Note:** For edits with multi-position anchors (e.g., cyclization with positions [0, 5]), both residues contribute to the composition.

#### B3. Anchor Local Context (9 features)

Features capturing physicochemical properties of anchor residues.

| Feature Name | Source Field | Computation | Description |
|-------------|-------------|-------------|-------------|
| `anchor_hydrophobic_frac` | `anchor_residues` | Fraction with I, L, V, A, F, W, M | Hydrophobic residues at anchors |
| `anchor_charged_frac` | `anchor_residues` | Fraction with D, E, K, R, H | Charged residues at anchors |
| `anchor_polar_frac` | `anchor_residues` | Fraction with S, T, N, Q, Y, C | Polar residues at anchors |
| `anchor_aromatic_frac` | `anchor_residues` | Fraction with F, Y, W | Aromatic residues at anchors |
| `anchor_small_frac` | `anchor_residues` | Fraction with G, A, S | Small residues at anchors |
| `anchor_proline_frac` | `anchor_residues` | Fraction with P | Proline at anchors (structural) |
| `anchor_terminal_frac` | `edits` | Fraction with terminal edit_family | Terminal modifications |
| `anchor_n_terminal_count` | `edits` | Count of n_terminal family | N-terminal edit count |
| `anchor_c_terminal_count` | `edits` | Count of c_terminal family | C-terminal edit count |

---

### C. Attachment-Aware Multi-Edit Features (28 features)

These features capture edit type, family structure, and multi-edit patterns.

#### C1. Edit Family Distribution (7 features)

| Feature Name | Source Field | Computation | Description |
|-------------|-------------|-------------|-------------|
| `edit_family_n_terminal` | `edits[*].edit_family` | Count | N-terminal modifications |
| `edit_family_c_terminal` | `edits[*].edit_family` | Count | C-terminal modifications |
| `edit_family_sidechain` | `edits[*].edit_family` | Count | Sidechain modifications |
| `edit_family_backbone` | `edits[*].edit_family` | Count | Backbone modifications |
| `edit_family_cyclization` | `edits[*].edit_family` | Count | Cyclization edits |
| `edit_family_substitution` | `edits[*].edit_family` | Count | Substitution edits |
| `edit_family_other` | `edits[*].edit_family` | Count | Other modifications |

#### C2. Edit Type Diversity (5 features)

| Feature Name | Source Field | Computation | Description |
|-------------|-------------|-------------|-------------|
| `edit_type_count_unique` | `edits[*].edit_type` | Count of unique types | Edit type diversity |
| `edit_type_entropy` | `edits[*].edit_type` | Shannon entropy | Edit type diversity measure |
| `attachment_type_count_unique` | `edits[*].attachment_semantics` | Count of unique attachment types | Attachment diversity |
| `has_cyclization` | `edits[*].edit_family` | Boolean (1/0) | Presence of cyclization |
| `has_backbone_mod` | `edits[*].edit_family` | Boolean (1/0) | Presence of backbone mod |

#### C3. Specific Edit Type Counts (10 features)

| Feature Name | Source Field | Computation | Description |
|-------------|-------------|-------------|-------------|
| `edit_n_methylation` | `edits[*].edit_type` | Count | N-methylation count |
| `edit_d_amino_acid` | `edits[*].edit_type` | Count | D-amino acid count |
| `edit_non_standard_aa` | `edits[*].edit_type` | Count | Non-standard AA count |
| `edit_head_to_tail_cyclization` | `edits[*].edit_type` | Count | Head-to-tail cyclization |
| `edit_sidechain_cyclization` | `edits[*].edit_type` | Count | Sidechain cyclization |
| `edit_disulfide` | `edits[*].edit_type` | Count | Disulfide bonds |
| `edit_acetylation` | `edits[*].edit_type` | Count | Acetylation |
| `edit_amidation` | `edits[*].edit_type` | Count | Amidation |
| `edit_phosphorylation` | `edits[*].edit_type` | Count | Phosphorylation |
| `edit_other_types` | `edits[*].edit_type` | Count of other types | Other modifications |

#### C4. Multi-Edit Aggregation (6 features)

| Feature Name | Source Field | Computation | Description |
|-------------|-------------|-------------|-------------|
| `anchor_pair_count` | `edits[*].anchor_positions` | Count with len > 1 | Paired/bridging edits |
| `cyclization_ring_size` | `edits[cyclization].edit_metadata` | Extract ring size if available | Cyclization ring size |
| `edit_anchor_overlap` | `edits[*].anchor_positions` | Count of shared positions | Overlapping anchor sites |
| `sequence_length` | `len(sequence)` | Count | Peptide sequence length |
| `sequence_cyclic` | `edits[*].edit_family` | Boolean (1/0) | Is peptide cyclic |
| `modification_rate` | `num_edits / len(sequence)` | Ratio | Edits per residue |

---

## Aggregation Strategy

### For Single-Position Anchors
- Use position and residue directly

### For Multi-Position Anchors (e.g., Cyclization)
- **Positions:** Contribute both positions to statistics (mean, std, range)
- **Residues:** Contribute both residues to composition features
- **Count:** Count as 1 edit but mark multi-anchor status

### For Global Edits (anchor_kind = "global")
- **Positions:** Excluded from position statistics
- **Residues:** Excluded from residue composition
- **Count:** Included in edit counts

### Missing or Invalid SMILES
- All chemistry descriptors set to 0.0
- Other features computed normally from edit structure

---

## Feature Vector Dimension

**Total Features:** 73

- A. Global chemistry: 10
- B. Anchor-aware local: 35
- C. Attachment-aware multi-edit: 28

---

## Implementation Notes

1. **Feature Scaling:** All features will be used as-is by XGBoost (tree-based models are scale-invariant)
2. **Missing Values:** Encoded as 0.0 for counts, mean for continuous features
3. **Categorical Encoding:** All categorical features are count-based or one-hot encoded
4. **Reproducibility:** Feature extraction is deterministic given the PEM schema

---

## Comparison to Baseline

### vs. Descriptor-Only Baseline
- **Shared:** Global chemistry features (10)
- **Added:** Anchor-aware (35) + Attachment-aware (28) features

### vs. Composition Baseline
- **Shared:** Sequence-level counts
- **Added:** Chemistry descriptors (10) + Anchor-aware (35) + Attachment-aware (28)

---

## Expected Signal

### Strong Predictors (Hypothesis)
1. `logp`, `tpsa` - Membrane permeability correlates with lipophilicity and polar surface area
2. `anchor_density` - Modification extent may affect permeability
3. `has_cyclization`, `sequence_cyclic` - Cyclization known to improve membrane permeability
4. `edit_family_backbone` - Backbone modifications (N-methylation, D-AA) reduce H-bonding
5. `anchor_hydrophobic_frac` - Hydrophobic anchors may correlate with permeability

### Mechanism Check Targets
- **Wrong-anchor control:** Shuffling `anchor_positions` and `anchor_residues` should degrade performance if anchor-awareness matters
- **Coarse-position control:** Randomizing positions within coarse bins should degrade performance if fine-grained position matters

---

## Validation

Feature extraction correctness will be validated by:
1. Checking feature distribution statistics (no NaN, reasonable ranges)
2. Verifying feature counts match schema (73 features per sample)
3. Comparing a subset of samples with manual calculation
4. Ensuring anchor residue compositions sum to 1.0 (normalized frequencies)

---

**Schema Version:** 1.0.0
**Author:** PEM Project
**Date:** April 4, 2026
