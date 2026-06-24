# PEM Unified Sample Schema Specification

**Version**: 1.0.0
**Date**: 2026-04-02
**Stage**: 2 - Data Processing

## Overview

This document specifies the standardized schema for representing peptide samples with chemical edits in the PEM project. The schema explicitly separates sequence backbone from chemical modifications, preserves anchor information, and maintains full assay provenance.

## Design Principles

1. **Separation of Concerns**: Sequence backbone vs. chemical edits are distinct
2. **Explicit Anchors**: All edit positions are explicitly tracked
3. **Assay Provenance**: Complete experimental context preserved
4. **Extensible Edits**: Support multiple edits per sample
5. **No Information Loss**: Avoid collapsing into single representation (e.g., SMILES)
6. **Validation-Friendly**: Schema supports strict type checking
7. **Parser Transparency**: Track parsing status and rules applied

## Top-Level Sample Schema

```python
{
  "sample_id": str,              # Unique identifier (dataset + local ID)
  "dataset": str,                # Source dataset name
  "sequence": str,               # Backbone sequence (canonical amino acids)
  "label": float,                # Primary experimental measurement
  "label_type": str,             # Type of measurement (e.g., "log_permeability")
  "label_unit": str,             # Unit of measurement
  "assay_type": str,             # Specific assay (e.g., "PAMPA")
  "assay_metadata": dict,        # Full experimental conditions
  "edits": List[Edit],           # Chemical modifications/edits
  "anchor_status": str,          # Overall anchor resolvability
  "provenance": dict,            # Source file, parsing info
  "split_metadata": dict,        # Train/val/test assignment (optional)
  "quality_flags": List[str]     # Any quality concerns
}
```

### Field Specifications

#### `sample_id` (required)
- **Type**: `str`
- **Format**: `{dataset_prefix}_{local_id}`
- **Example**: `"CYCPEPT_001234"`, `"PEPMSND_000567"`
- **Uniqueness**: Must be globally unique across all datasets
- **Validation**: Non-empty, matches pattern `^[A-Z]+_[0-9]+$`

#### `dataset` (required)
- **Type**: `str`
- **Values**: `"CycPeptMPDB_PAMPA"`, `"PepMSND"`, `"DBAASP"`
- **Purpose**: Source dataset identification
- **Validation**: Must be one of allowed dataset names

#### `sequence` (required)
- **Type**: `str`
- **Content**: Canonical amino acid sequence (backbone only)
- **Alphabet**: `ACDEFGHIKLMNPQRSTVWY` (20 standard amino acids)
- **Modifications**: NOT included in sequence (use `edits` instead)
- **Cyclization**: NOT encoded in sequence (use `edits` with `edit_family="cyclization"`)
- **Example**: `"ACDEFGHIKLM"` (linear sequence, no modification notation)
- **Validation**:
  - Non-empty
  - Contains only standard amino acids
  - Length between 3 and 100 residues (configurable)
- **Rationale**: Clean separation allows anchor-based modeling without parsing complexity

#### `label` (required)
- **Type**: `float`
- **Content**: Primary experimental measurement value
- **Example**: `-4.52` (log permeability), `120.5` (half-life in minutes)
- **Validation**: Finite numeric value (no NaN, no Inf)

#### `label_type` (required)
- **Type**: `str`
- **Values**: `"log_permeability"`, `"half_life_minutes"`, `"log_MIC"`, etc.
- **Purpose**: Semantic meaning of label
- **Validation**: Non-empty, standardized vocabulary

#### `label_unit` (required)
- **Type**: `str`
- **Values**: `"log(cm/s)"`, `"minutes"`, `"log(μM)"`, etc.
- **Purpose**: Physical unit of measurement
- **Validation**: Non-empty string

#### `assay_type` (required)
- **Type**: `str`
- **Values**: `"PAMPA"`, `"blood_stability"`, `"MIC_E.coli"`, etc.
- **Purpose**: Specific experimental assay
- **Important**: Different assay types should NOT be merged (e.g., PAMPA vs. Caco-2)
- **Validation**: Non-empty, standardized per dataset

#### `assay_metadata` (required)
- **Type**: `dict[str, Any]`
- **Content**: Complete experimental conditions
- **Example**:
  ```python
  {
    "temperature_celsius": 37.0,
    "pH": 7.4,
    "buffer": "PBS",
    "incubation_time_hours": 2.0,
    "organism": "E. coli",  # For antimicrobial assays
    "donor_species": "human",  # For stability assays
    "membrane_type": "PAMPA"  # For permeability
  }
  ```
- **Validation**: Non-empty dict, required keys vary by assay type

#### `edits` (required, can be empty list)
- **Type**: `List[Edit]`
- **Content**: All chemical modifications to the backbone sequence
- **Example**: `[Edit(...), Edit(...)]` or `[]` for unmodified peptides
- **Validation**: List of valid Edit objects (see Edit Schema below)

#### `anchor_status` (required)
- **Type**: `str`
- **Values**: `"explicit_anchor"`, `"weakly_inferable"`, `"not_resolvable"`, `"no_edits"`
- **Purpose**: Overall anchor resolvability for this sample
- **Derivation**: From Stage 1 census classification
- **Validation**: Must be one of allowed values

#### `provenance` (required)
- **Type**: `dict`
- **Content**: Source file and parsing information
- **Example**:
  ```python
  {
    "source_file": "cycpeptmpdb_pampa.csv",
    "source_row_index": 1234,
    "raw_sequence_notation": "Ac-C[Cys]DEFG-NH2",
    "raw_modification_field": "N-acetylation, C-amidation",
    "parser_version": "1.0.0",
    "parsing_date": "2026-04-02T14:30:00",
    "census_report": "cycpeptmpdb_pampa_census.json"
  }
  ```
- **Validation**: Must contain `source_file`, `source_row_index`, `parser_version`

#### `split_metadata` (optional)
- **Type**: `dict` or `null`
- **Content**: Train/val/test assignment
- **Example**:
  ```python
  {
    "split": "train",  # "train", "val", "test"
    "fold": 0,  # For cross-validation
    "split_strategy": "random",  # "random", "scaffold", "time"
    "random_seed": 42
  }
  ```
- **Validation**: If present, must contain `split` key

#### `quality_flags` (required)
- **Type**: `List[str]`
- **Content**: Any quality concerns or warnings
- **Example**: `["weak_anchor_inference", "unusual_sequence_length", "outlier_label"]`
- **Validation**: List of strings (can be empty)

---

## Edit Schema

Each chemical modification is represented as an Edit object with detailed structural information.

```python
{
  "edit_id": str,                    # Unique identifier for this edit
  "edit_family": str,                # High-level category
  "edit_type": str,                  # Specific modification type
  "anchor_kind": str,                # Anchor classification
  "anchor_positions": List[int],     # 0-indexed positions in sequence
  "anchor_residues": List[str],      # Residues at anchor positions
  "chem_rep_raw": str,               # Original notation from source
  "chem_rep_canonical": str,         # Standardized representation
  "attachment_semantics": str,       # Where/how modification attaches
  "parser_status": str,              # Parsing success/failure
  "rule_id": str,                    # Parsing rule that matched
  "edit_metadata": dict              # Additional edit-specific info
}
```

### Edit Field Specifications

#### `edit_id` (required)
- **Type**: `str`
- **Format**: `{sample_id}_edit_{index}`
- **Example**: `"CYCPEPT_001234_edit_0"`
- **Uniqueness**: Unique within the sample
- **Validation**: Non-empty, matches pattern

#### `edit_family` (required)
- **Type**: `str`
- **Values**: Standardized vocabulary
  - `"n_terminal"` - N-terminal modification
  - `"c_terminal"` - C-terminal modification
  - `"sidechain"` - Side chain modification
  - `"backbone"` - Backbone modification (e.g., N-methylation)
  - `"cyclization"` - Cyclization constraint
  - `"substitution"` - Residue substitution (including D-amino acids)
  - `"other"` - Other modifications
- **Purpose**: High-level categorization for anchor modeling
- **Validation**: Must be one of allowed values

#### `edit_type` (required)
- **Type**: `str`
- **Values**: Specific modification name (extensible vocabulary)
  - Examples: `"acetylation"`, `"amidation"`, `"phosphorylation"`, `"methylation"`, `"disulfide_bridge"`, `"head_to_tail_cyclization"`, `"d_amino_acid"`, `"n_methylation"`
- **Purpose**: Detailed chemical modification type
- **Validation**: Non-empty string

#### `anchor_kind` (required)
- **Type**: `str`
- **Values**: `"explicit"`, `"inferred"`, `"ambiguous"`, `"global"`
- **Definitions**:
  - `"explicit"`: Anchor position directly annotated in source data
  - `"inferred"`: Position inferred by parser (flagged as inference)
  - `"ambiguous"`: Multiple possible positions, unclear which is correct
  - `"global"`: Modification affects entire sequence (e.g., some cyclizations)
- **Purpose**: Transparency about anchor confidence
- **Validation**: Must be one of allowed values

#### `anchor_positions` (required)
- **Type**: `List[int]`
- **Content**: 0-indexed positions in the backbone sequence
- **Example**: `[0]` for N-terminal, `[5]` for position 6, `[3, 7]` for disulfide bridge
- **Empty**: `[]` for global modifications (e.g., cyclization without specific anchor)
- **Validation**:
  - All positions must be valid indices in sequence
  - Positions must be sorted
  - No duplicates

#### `anchor_residues` (required)
- **Type**: `List[str]`
- **Content**: Amino acids at anchor positions
- **Example**: `["M"]` for Met, `["C", "C"]` for Cys-Cys bridge
- **Empty**: `[]` for terminal or global modifications
- **Validation**:
  - Length must match `anchor_positions` length
  - All must be valid amino acids

#### `chem_rep_raw` (required)
- **Type**: `str`
- **Content**: Original notation from source data
- **Example**: `"Ac-"`, `"[pS]"`, `"cyclic(1-12)"`
- **Purpose**: Preserve original representation for audit trail
- **Validation**: Non-empty string

#### `chem_rep_canonical` (required)
- **Type**: `str`
- **Content**: Standardized representation (HELM, SMILES, or custom)
- **Example**: `"PEPTIDE1{A.C.D.E.F}$$$$"` (HELM), `"CC(=O)N-"` (SMILES for Acetyl)
- **Purpose**: Canonical form for comparison and modeling
- **Validation**: Non-empty string, valid format

#### `attachment_semantics` (required)
- **Type**: `str`
- **Values**: Where/how modification attaches
  - `"n_terminus"` - Attaches to N-terminus
  - `"c_terminus"` - Attaches to C-terminus
  - `"sidechain_oxygen"` - O-linked (Ser, Thr, Tyr)
  - `"sidechain_nitrogen"` - N-linked (Lys)
  - `"sidechain_sulfur"` - S-linked (Cys)
  - `"backbone_nitrogen"` - Backbone N-methylation
  - `"bridge"` - Connects multiple positions
  - `"global"` - Whole-peptide constraint
- **Purpose**: Specify attachment chemistry
- **Validation**: Non-empty string

#### `parser_status` (required)
- **Type**: `str`
- **Values**: `"success"`, `"inferred"`, `"partial"`, `"failed"`
- **Definitions**:
  - `"success"`: Parsed with high confidence
  - `"inferred"`: Required heuristic inference
  - `"partial"`: Some information missing or ambiguous
  - `"failed"`: Could not parse reliably
- **Purpose**: Transparency about parsing confidence
- **Validation**: Must be one of allowed values

#### `rule_id` (required)
- **Type**: `str`
- **Content**: Identifier of parsing rule that matched
- **Example**: `"cycpept_acetylation_v1"`, `"generic_terminal_mod_v1"`
- **Purpose**: Track which parsing logic was used (for debugging and versioning)
- **Validation**: Non-empty string

#### `edit_metadata` (required)
- **Type**: `dict[str, Any]`
- **Content**: Additional edit-specific information
- **Example**:
  ```python
  {
    "stereo": "D",  # For D-amino acids
    "bridge_partner": 7,  # For disulfide bridges
    "confidence_score": 0.85,
    "alternative_interpretations": [...]
  }
  ```
- **Validation**: Can be empty dict `{}`

---

## Special Cases

### 1. Cyclic Peptides

**Representation**: Cyclization is an **edit**, not a sequence property.

**Example**:
```python
{
  "sequence": "ACDEFGHIKLM",  # Linear sequence representation
  "edits": [
    {
      "edit_id": "CYCPEPT_001_edit_0",
      "edit_family": "cyclization",
      "edit_type": "head_to_tail_cyclization",
      "anchor_kind": "explicit",
      "anchor_positions": [0, 10],  # N-terminus to C-terminus
      "anchor_residues": ["A", "M"],
      "chem_rep_raw": "cyclic(1-11)",
      "chem_rep_canonical": "PEPTIDE1{[A.C.D.E.F.G.H.I.K.L.M] |$PEPTIDE1,PEPTIDE1,1:R3-11:R3$|}",
      "attachment_semantics": "bridge",
      "parser_status": "success",
      "rule_id": "cycpept_head_to_tail_v1",
      "edit_metadata": {
        "cyclization_type": "head_to_tail",
        "ring_size": 11
      }
    }
  ]
}
```

**Rationale**:
- Allows anchor-based modeling of cyclization
- Preserves linear sequence for comparison
- Explicit about cyclization chemistry

### 2. Disulfide Bridges

**Representation**: Bridge between two Cys residues.

**Example**:
```python
{
  "edit_id": "CYCPEPT_002_edit_0",
  "edit_family": "cyclization",
  "edit_type": "disulfide_bridge",
  "anchor_kind": "explicit",
  "anchor_positions": [2, 8],  # Cys at positions 2 and 8
  "anchor_residues": ["C", "C"],
  "chem_rep_raw": "disulfide(3-9)",  # 1-indexed in source
  "chem_rep_canonical": "C-S-S-C",
  "attachment_semantics": "bridge",
  "parser_status": "success",
  "rule_id": "disulfide_v1",
  "edit_metadata": {
    "bridge_type": "disulfide",
    "bridge_length": 6
  }
}
```

### 3. D-Amino Acids

**Representation**: Substitution edit at specific position.

**Example**:
```python
{
  "sequence": "ACDEFGHIKLM",  # L-amino acids (canonical)
  "edits": [
    {
      "edit_id": "PEPMSND_003_edit_0",
      "edit_family": "substitution",
      "edit_type": "d_amino_acid",
      "anchor_kind": "explicit",
      "anchor_positions": [5],  # Position 5 is D-Gly
      "anchor_residues": ["G"],
      "chem_rep_raw": "dG",
      "chem_rep_canonical": "[dG]",
      "attachment_semantics": "backbone",
      "parser_status": "success",
      "rule_id": "d_amino_acid_v1",
      "edit_metadata": {
        "stereo": "D",
        "original_residue": "G"
      }
    }
  ]
}
```

### 4. Multiple Edits

**Representation**: List of edits, each with own anchor.

**Example**:
```python
{
  "sequence": "ACDEFGHIKLM",
  "edits": [
    {
      "edit_id": "SAMPLE_004_edit_0",
      "edit_family": "n_terminal",
      "edit_type": "acetylation",
      "anchor_kind": "explicit",
      "anchor_positions": [0],
      "anchor_residues": ["A"],
      "chem_rep_raw": "Ac-",
      "chem_rep_canonical": "CC(=O)-",
      "attachment_semantics": "n_terminus",
      "parser_status": "success",
      "rule_id": "n_term_acetyl_v1",
      "edit_metadata": {}
    },
    {
      "edit_id": "SAMPLE_004_edit_1",
      "edit_family": "c_terminal",
      "edit_type": "amidation",
      "anchor_kind": "explicit",
      "anchor_positions": [10],
      "anchor_residues": ["M"],
      "chem_rep_raw": "-NH2",
      "chem_rep_canonical": "-NH2",
      "attachment_semantics": "c_terminus",
      "parser_status": "success",
      "rule_id": "c_term_amide_v1",
      "edit_metadata": {}
    },
    {
      "edit_id": "SAMPLE_004_edit_2",
      "edit_family": "sidechain",
      "edit_type": "phosphorylation",
      "anchor_kind": "inferred",
      "anchor_positions": [3],
      "anchor_residues": ["S"],  # Assumed Ser in original notation
      "chem_rep_raw": "pS",
      "chem_rep_canonical": "[phosphoS]",
      "attachment_semantics": "sidechain_oxygen",
      "parser_status": "inferred",
      "rule_id": "phospho_inference_v1",
      "edit_metadata": {
        "inference_confidence": 0.9
      }
    }
  ]
}
```

### 5. No Edits (Unmodified Peptide)

**Representation**: Empty edit list.

**Example**:
```python
{
  "sequence": "ACDEFGHIKLM",
  "edits": [],
  "anchor_status": "no_edits"
}
```

---

## Controlled Vocabularies

### Edit Families
- `n_terminal`
- `c_terminal`
- `sidechain`
- `backbone`
- `cyclization`
- `substitution`
- `other`

### Edit Types (Extensible)
Common types (can be extended as needed):
- **N-terminal**: `acetylation`, `formylation`, `myristoylation`
- **C-terminal**: `amidation`, `methylation`
- **Sidechain**: `phosphorylation`, `methylation`, `acetylation`, `glycosylation`
- **Backbone**: `n_methylation`
- **Cyclization**: `head_to_tail_cyclization`, `disulfide_bridge`, `sidechain_cyclization`
- **Substitution**: `d_amino_acid`, `unnatural_amino_acid`

### Anchor Kinds
- `explicit` - Directly annotated
- `inferred` - Parser inference
- `ambiguous` - Multiple possibilities
- `global` - No specific anchor

### Attachment Semantics
- `n_terminus`
- `c_terminus`
- `sidechain_oxygen`
- `sidechain_nitrogen`
- `sidechain_sulfur`
- `backbone_nitrogen`
- `bridge`
- `global`

### Parser Status
- `success` - High confidence
- `inferred` - Required inference
- `partial` - Incomplete information
- `failed` - Could not parse

---

## Serialization Formats

### 1. JSONL (JSON Lines)

**File format**: One JSON object per line

**Example**: `data/processed/pem_schema/cycpeptmpdb_pampa.jsonl`

```jsonl
{"sample_id": "CYCPEPT_001", "dataset": "CycPeptMPDB_PAMPA", "sequence": "ACDEFG", ...}
{"sample_id": "CYCPEPT_002", "dataset": "CycPeptMPDB_PAMPA", "sequence": "GHIKLM", ...}
```

**Benefits**:
- Human-readable
- Streamable
- Easy to inspect
- Good for logging

### 2. Parquet

**File format**: Columnar Apache Parquet

**Example**: `data/processed/pem_schema/cycpeptmpdb_pampa.parquet`

**Benefits**:
- Efficient storage
- Fast loading
- Good for ML pipelines
- Schema enforcement

**Note**: Nested structures (edits, metadata) stored as JSON strings in Parquet

---

## Validation Rules

### Sample-Level Validation

1. **Required Fields**: All required fields must be present
2. **Type Checking**: All fields must match specified types
3. **Unique ID**: `sample_id` must be unique within dataset
4. **Sequence Validation**:
   - Only standard amino acids
   - Length within allowed range
5. **Label Validation**: Finite numeric value
6. **Edit Consistency**: All edits must reference valid sequence positions
7. **Anchor Status Consistency**: Must match edit anchor classifications

### Edit-Level Validation

1. **Required Fields**: All edit fields required
2. **Position Validity**: `anchor_positions` must be valid sequence indices
3. **Residue Consistency**: `anchor_residues` must match sequence at `anchor_positions`
4. **Vocabulary**: `edit_family`, `anchor_kind`, `parser_status` from controlled vocabularies
5. **Anchor-Position Alignment**: Length of `anchor_positions` must match `anchor_residues`

---

## Version History

**v1.0.0** (2026-04-02)
- Initial schema specification
- Support for CycPeptMPDB, PepMSND, DBAASP
- Explicit anchor tracking
- Extensible edit representation
- Cyclic peptide support

---

## References

- Stage 1 Census: `STAGE1_CENSUS.md`
- Stage 2 Implementation: `STAGE2_PROCESSING.md`
- Validation Code: `src/data/pem_schema.py`
- Converters: `src/data/converters/`
