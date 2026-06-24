# PEM Chemical Representation Specification v1.0

**Version**: 1.0.0
**Date**: 2026-04-02
**Stage**: 3 - Attachment-Aware Chemical Edit Representation
**Scope**: Conservative, minimal but rigorous

## Philosophy

### Conservative Scope (v1)
- **DO**: Support well-defined, unambiguous edit families
- **DO**: Maintain attachment context explicitly
- **DO**: Enable reproducible canonicalization
- **DO NOT**: Attempt full generality for all PTMs
- **DO NOT**: Handle complex polymer modifications
- **DO NOT**: Silently normalize or guess structures

### Audit-First Design
- Every parsing decision tracked with rule_id
- Every failure logged with reason
- Every canonicalization step documented
- No silent fallbacks or assumptions

## Covered Edit Families (v1)

### 1. Non-Canonical Amino Acid (ncAA) Substitution

**Definition**: Replacement of standard amino acid with non-canonical variant

**Supported Cases**:
- D-amino acids (stereo inversion)
- N-methylated amino acids
- Other well-characterized ncAAs with known structure

**Examples**:
- D-Alanine replacing L-Alanine
- N-methyl-Glycine replacing Glycine
- Norleucine replacing Leucine

**Exclusions**:
- Novel/poorly characterized ncAAs
- Ambiguous stereochemistry
- Missing structural information

### 2. Cyclization / Linker-Like Relations

**Definition**: Covalent bonds creating cycles or bridges

**Supported Cases**:
- Disulfide bridges (Cys-Cys)
- Head-to-tail cyclization (explicit positions)
- Lactam bridges (explicit positions)

**Examples**:
- Disulfide: Cys3-Cys8
- Head-to-tail: N-terminus to C-terminus
- Lactam: Lys-Asp bridge

**Exclusions**:
- Implicit/ambiguous cyclization
- Complex multi-point cycles
- Unclear attachment points

### 3. Simple Side-Chain Modifications

**Definition**: Well-characterized chemical groups attached to side chains

**Supported Cases**:
- Acetylation (Lys)
- Phosphorylation (Ser, Thr, Tyr)
- Methylation (Lys, Arg)
- Amidation (C-terminus)

**Examples**:
- Acetyl-Lys at position 5
- Phospho-Ser at position 3
- N-terminal acetylation

**Exclusions**:
- Complex glycosylation
- Large chemical scaffolds
- Ambiguous attachment sites
- Multiple modification sites per residue

## Not Covered (v1 Exclusions)

### Explicitly Out of Scope:
- Complex glycosylation patterns
- Lipidation with long chains
- PEGylation
- Multiple overlapping modifications
- Polymer attachments
- Novel/uncharacterized modifications
- Modifications without clear attachment semantics

## Attachment-Aware Representation

### Core Principle
**Chemical edit = Moiety + Attachment + Context**

Every edit representation must include:
1. **Moiety**: The chemical group being added/changed
2. **Attachment**: Where and how it connects
3. **Context**: Host-side anchor information
4. **Rule**: Which parsing/canonicalization rule was applied

### Representation Components

```python
ChemRepr:
  # Moiety
  moiety_smiles: str              # SMILES of chemical group
  moiety_name: str                # Standard name

  # Attachment
  attachment_atom: str            # Attachment atom type
  attachment_bond: str            # Bond type
  attachment_position: int        # Position in sequence
  attachment_residue: str         # Host residue

  # Context
  context_before: str             # Upstream sequence context
  context_after: str              # Downstream sequence context
  context_window: int             # Window size (default: 3)

  # Parsing metadata
  rule_id: str                    # Canonicalization rule
  parser_status: str              # success/inferred/failed
  canonicalization_notes: str     # Any transformations applied
  exclusion_reason: Optional[str] # If failed
```

## Representation Formats

### Format 1: Tagged Chemical String (TCS)

**Syntax**: `<moiety>@<attachment>:[context]`

**Components**:
- `<moiety>`: SMILES or standard abbreviation
- `<attachment>`: `{residue}{position}.{atom}.{bond}`
- `[context]`: `{before}-{residue}-{after}`

**Examples**:

```
# D-Alanine at position 5
dAla@A5.CA.single:[CD-A-EF]

# Acetyl-Lysine at position 3
Ac@K3.NZ.single:[AC-K-DE]

# Phospho-Serine at position 7
PO3@S7.OG.single:[FG-S-HI]

# Disulfide bridge Cys3-Cys8
S-S@C3.SG.single-C8.SG.single:[AB-C-DE]~[GH-C-IJ]
```

**Format Specification**:
```
TCS := <moiety>@<attachment>:[context]

<moiety> := SMILES | standard_name
<attachment> := <single_attach> | <bridge_attach>
<single_attach> := {AA}{pos}.{atom}.{bond}
<bridge_attach> := {AA}{pos1}.{atom1}.{bond1}-{AA}{pos2}.{atom2}.{bond2}
<context> := [{upstream}-{anchor}-{downstream}]
           | [{upstream1}-{anchor1}-{downstream1}]~[{upstream2}-{anchor2}-{downstream2}]

{AA} := single-letter amino acid code
{pos} := integer (0-indexed)
{atom} := atom name (CA, CB, NZ, OG, SG, etc.)
{bond} := single | double | triple
{upstream} := sequence context before
{anchor} := anchor residue
{downstream} := sequence context after
```

**Validation Rules**:
- `{pos}` must be valid sequence index
- `{AA}` must match sequence at position
- `{atom}` must be valid for residue type
- Context must match actual sequence

### Format 2: Attachment-Aware Fingerprint (AAF)

**Concept**: Concatenated fingerprints preserving attachment information

**Components**:
```python
AAF = [
  moiety_fp,          # Molecular fingerprint of chemical group
  attachment_fp,      # Encoding of attachment point
  context_fp          # Fingerprint of sequence context
]
```

**Implementation**:

```python
# Moiety fingerprint (Morgan, radius=2, 2048 bits)
moiety_fp = rdkit.GetMorganFingerprintAsBitVect(moiety_mol, 2, 2048)

# Attachment fingerprint (one-hot + categorical)
attachment_fp = [
  residue_one_hot,     # 20-dim (standard AAs)
  atom_type_one_hot,   # e.g., [N, O, S, C] -> 4-dim
  bond_type_one_hot    # [single, double, triple] -> 3-dim
]

# Context fingerprint (sequence embedding or AAC)
context_fp = [
  amino_acid_composition,  # 20-dim
  position_encoding        # relative position
]

# Final concatenation
AAF = concat(moiety_fp, attachment_fp, context_fp)
# Dimension: 2048 + 27 + 20 + pos_dim
```

**Properties**:
- Fixed-length vector
- Preserves attachment information
- Allows distance metrics
- Suitable for ML models

## Canonicalization Rules

### Rule Versioning

Each parsing rule has unique identifier: `{family}_{type}_v{version}`

**Examples**:
- `ncaa_d_amino_acid_v1`
- `cyclization_disulfide_v1`
- `sidechain_acetylation_v1`

### Standard Rules (v1.0)

#### Rule: `ncaa_d_amino_acid_v1`

**Input**: D-amino acid notation (e.g., "dAla", "D-Ala")

**Canonicalization**:
1. Extract amino acid type
2. Verify position in sequence
3. Generate SMILES with inverted stereochemistry
4. Create TCS: `d{AA}@{AA}{pos}.CA.single:[context]`

**Validation**:
- Position must be valid
- Residue must match sequence
- Stereochemistry must be explicit

**Example**:
```
Input: "dAla" at position 5 in sequence "ACDEFG..."
Output: "dAla@A5.CA.single:[CD-A-EF]"
Rule: ncaa_d_amino_acid_v1
Status: success
```

#### Rule: `cyclization_disulfide_v1`

**Input**: Disulfide bridge notation (e.g., "Cys3-Cys8")

**Canonicalization**:
1. Extract positions
2. Verify both are Cys
3. Order by position (lower first)
4. Create TCS: `S-S@C{pos1}.SG.single-C{pos2}.SG.single:[ctx1]~[ctx2]`

**Validation**:
- Both positions must be Cys
- Positions must be different
- Positions must be valid

**Example**:
```
Input: Disulfide between Cys3 and Cys8
Sequence: "ABCDEFGHIJ" where C at 3,8
Output: "S-S@C3.SG.single-C8.SG.single:[AB-C-DE]~[FG-C-HI]"
Rule: cyclization_disulfide_v1
Status: success
```

#### Rule: `sidechain_acetylation_v1`

**Input**: Acetylation notation (e.g., "Ac-Lys", "AcK")

**Canonicalization**:
1. Extract position and residue
2. Verify acetylatable (Lys, N-term)
3. Generate acetyl SMILES: `CC(=O)`
4. Create TCS: `Ac@{AA}{pos}.NZ.single:[context]`

**Validation**:
- Residue must be Lys (or N-term)
- Position must be valid

**Example**:
```
Input: Acetyl-Lys at position 3
Sequence: "ACKDEFG..."
Output: "Ac@K3.NZ.single:[AC-K-DE]"
Rule: sidechain_acetylation_v1
Status: success
```

#### Rule: `sidechain_phosphorylation_v1`

**Input**: Phosphorylation notation (e.g., "pSer", "Phospho-Ser")

**Canonicalization**:
1. Extract position and residue
2. Verify phosphorylatable (Ser, Thr, Tyr)
3. Generate phosphate SMILES: `OP(=O)(O)O`
4. Create TCS: `PO3@{AA}{pos}.OG.single:[context]`

**Validation**:
- Residue must be Ser/Thr/Tyr
- Position must be valid

**Example**:
```
Input: Phospho-Ser at position 7
Sequence: "ABCDEFSHIJ..."
Output: "PO3@S7.OG.single:[FG-S-HI]"
Rule: sidechain_phosphorylation_v1
Status: success
```

## Parser Status Values

### `success`
- Unambiguous parsing
- All validation passed
- High confidence representation

### `inferred`
- Required heuristic inference
- Assumptions documented in notes
- Medium confidence

### `partial`
- Some information missing
- Incomplete representation
- Low confidence

### `failed`
- Cannot parse reliably
- Must log exclusion_reason

## Exclusion Reasons

### Standardized Categories

1. `ambiguous_structure` - Chemical structure unclear
2. `missing_attachment` - Attachment point not specified
3. `invalid_position` - Position out of range or wrong residue
4. `unsupported_modification` - Not in v1 scope
5. `complex_pattern` - Too complex for v1 parser
6. `incomplete_data` - Required fields missing
7. `validation_failed` - Failed canonicalization checks

## Auditing Requirements

### For Each Edit Parsed

Must record:
- `input_raw`: Original notation
- `rule_id`: Which rule was applied
- `parser_status`: Success level
- `canonicalization_notes`: What transformations occurred
- `exclusion_reason`: If failed (null otherwise)

### Logs Must Include

- Total edits attempted
- Success count by rule
- Failure count by reason
- Examples of each status
- Coverage statistics

## Output Format

### ChemRepr JSON Schema

```json
{
  "edit_id": "SAMPLE_001_edit_0",
  "edit_family": "sidechain",
  "edit_type": "acetylation",

  "chem_repr": {
    "tcs": "Ac@K3.NZ.single:[AC-K-DE]",
    "moiety_smiles": "CC(=O)",
    "moiety_name": "acetyl",

    "attachment": {
      "position": 3,
      "residue": "K",
      "atom": "NZ",
      "bond": "single"
    },

    "context": {
      "before": "AC",
      "anchor": "K",
      "after": "DE",
      "window": 3
    },

    "fingerprint": {
      "aaf": [0.0, 1.0, ..., 0.0],  # Full AAF vector
      "dimension": 2095
    }
  },

  "parsing": {
    "rule_id": "sidechain_acetylation_v1",
    "parser_status": "success",
    "canonicalization_notes": "Standard acetyl group, Lys NZ attachment",
    "exclusion_reason": null
  },

  "provenance": {
    "input_raw": "AcK3",
    "parser_version": "1.0.0",
    "parsing_date": "2026-04-02T14:30:00"
  }
}
```

## Coverage Metrics

### Required Reporting

For each dataset, report:

1. **Total edit count**
2. **Covered edits** (by family)
   - ncAA: count, percentage
   - cyclization: count, percentage
   - sidechain: count, percentage
3. **Excluded edits** (by reason)
   - unsupported_modification: count
   - ambiguous_structure: count
   - etc.
4. **Parser status distribution**
   - success: count, percentage
   - inferred: count, percentage
   - failed: count, percentage
5. **Representative examples**
   - 3 examples per success status
   - 3 examples per failure reason

## Validation Protocol

### Canonicalization Validation

For each parsed edit:
1. Verify position is valid index
2. Verify residue matches sequence
3. Verify atom type valid for residue
4. Verify attachment chemistry makes sense
5. Verify context matches actual sequence

### Round-Trip Testing

For tagged representations:
1. Parse to internal representation
2. Canonicalize
3. Re-serialize
4. Compare to original
5. Document any differences

## Version History

**v1.0.0** (2026-04-02)
- Initial conservative scope
- Support for ncAA, cyclization, simple sidechain mods
- Tagged Chemical String (TCS) format
- Attachment-Aware Fingerprint (AAF) format
- Versioned canonicalization rules
- Comprehensive auditing

## Future Expansion (v2.0+)

**Possible additions** (not in v1):
- N-terminal/C-terminal modifications
- Glycosylation (simple)
- Additional ncAAs
- Methylation variants
- Additional cyclization types

**Principles for expansion**:
- Only add when well-characterized
- Maintain attachment-aware representation
- Preserve audit trail
- Version all new rules

## References

- Stage 2 Schema: `docs/schema_spec.md`
- PEM Schema Implementation: `src/data/pem_schema.py`
- Chemical Representation Code: `src/data/chem_repr/`
