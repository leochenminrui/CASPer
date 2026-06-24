#!/usr/bin/env python3
"""
Automated Line-by-Line Dataset Audit for CycPeptMPDB PAMPA.

Audits ALL 7,298 rows — no sampling, no manual spot-check.
For each row, classifies:
  1. Parse status (success / why failed)
  2. Anchor resolvability (explicit / inferred / ambiguous / global / no_edits)
  3. Edit family & type inventory
  4. Quality flags (missing SMILES, duplicate sequence, ...)

Output:
  results/benchmark/audit/
    per_sample_audit.csv          — one row per input sample (7,298 rows)
    excluded_samples.csv          — all excluded rows with reasons
    monomer_coverage.csv          — every monomer × count × mapped status
    audit_summary.json / .csv     — aggregate statistics
    audit_methods_text.md         — manuscript methods paragraph
"""

import sys, json, csv, ast, logging
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data.converters.cycpeptmpdb_converter import CycPeptMPDBConverter


def audit_all():
    df = pd.read_csv(PROJECT_ROOT / 'data/raw/cycpeptmpdb_pampa.csv', low_memory=False)
    conv = CycPeptMPDBConverter(strict_mode=True)

    output_dir = PROJECT_ROOT / 'results/benchmark/audit'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ═══ Step 1: Monomer census ═══════════════════════════════════════════
    logger.info("Step 1/4: Monomer coverage census...")
    monomer_counts = Counter()
    monomer_mapped = set()
    monomer_unmapped = Counter()

    for _, row in df.iterrows():
        seq_str = row.get('Sequence')
        if not seq_str or pd.isna(seq_str):
            continue
        try:
            monomers = ast.literal_eval(str(seq_str))
        except (ValueError, SyntaxError):
            continue
        if not isinstance(monomers, list):
            continue
        for m in monomers:
            monomer_counts[m] += 1
            if conv.monomer_to_aa.get(m):
                monomer_mapped.add(m)
            elif not m.startswith('Mono') and m not in ('glyco-', 'deca-', 'medl-'):
                monomer_unmapped[m] += 1

    # ═══ Step 2: Per-sample audit ═════════════════════════════════════════
    logger.info("Step 2/4: Per-sample audit (all 7,298 rows)...")

    seq_counter = Counter()
    for _, row in df.iterrows():
        s = row.get('Sequence')
        if s and pd.notna(s):
            seq_counter[str(s)] += 1

    audit_rows = []
    excluded_rows = []
    anchor_status_counts = Counter()
    edit_family_inventory = Counter()
    edit_type_inventory = Counter()
    quality_flag_counts = Counter()
    parse_failure_reasons = Counter()

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        csv_id = str(row_dict.get('ID', ''))
        sample_id = f"CYCPEPTMPDB_PAMPA_{idx+1:06d}"
        seq_str = str(row_dict.get('Sequence', ''))
        audit = {
            'audit_index': idx, 'csv_row': idx + 2,
            'csv_id': csv_id, 'sample_id': sample_id,
            'source': str(row_dict.get('Source', '')),
            'year': int(row_dict['Year']) if pd.notna(row_dict.get('Year')) else None,
            'molecule_shape': str(row_dict.get('Molecule_Shape', '')),
            'monomer_length': int(row_dict['Monomer_Length']) if pd.notna(row_dict.get('Monomer_Length')) else 0,
        }

        # Quality flags
        flags = []
        smiles = str(row_dict.get('SMILES', ''))
        if not smiles or pd.isna(row_dict.get('SMILES')):
            flags.append('no_smiles')
            quality_flag_counts['no_smiles'] += 1
        pampa = row_dict.get('PAMPA')
        if pampa is None or pd.isna(pampa):
            flags.append('no_pampa_label')
            quality_flag_counts['no_pampa_label'] += 1
        if seq_counter.get(seq_str, 0) > 1:
            flags.append('duplicate_sequence')
            quality_flag_counts['duplicate_sequence'] += 1
        if seq_counter.get(seq_str, 0) > 5:
            flags.append('highly_redundant_sequence')
        audit['quality_flags'] = ';'.join(flags) if flags else ''

        # —— Parse ——
        if not seq_str or pd.isna(seq_str):
            audit['parse_status'] = 'failed'
            audit['exclusion_reason'] = 'empty_sequence'
            audit['anchor_status'] = 'not_resolvable'
            audit['backbone_length'] = 0
            parse_failure_reasons['empty_sequence'] += 1
            excluded_rows.append(audit)
            audit_rows.append(audit)
            continue

        try:
            monomers = ast.literal_eval(seq_str)
        except (ValueError, SyntaxError):
            audit['parse_status'] = 'failed'
            audit['exclusion_reason'] = 'unparseable_monomer_list'
            audit['anchor_status'] = 'not_resolvable'
            audit['backbone_length'] = 0
            parse_failure_reasons['unparseable_list'] += 1
            excluded_rows.append(audit)
            audit_rows.append(audit)
            continue

        if not isinstance(monomers, list):
            audit['parse_status'] = 'failed'
            audit['exclusion_reason'] = 'not_a_list'
            audit['anchor_status'] = 'not_resolvable'
            audit['backbone_length'] = 0
            parse_failure_reasons['not_a_list'] += 1
            excluded_rows.append(audit)
            audit_rows.append(audit)
            continue

        # Build backbone
        backbone = []
        unknown = []
        for m in monomers:
            if m.startswith('ac-') or m.startswith('-') or m.endswith('-'):
                continue
            if m.startswith('Mono'):
                continue
            if m in ('glyco-', 'deca-', 'medl-'):
                continue
            aa = conv.monomer_to_aa.get(m)
            if aa:
                backbone.append(aa)
            else:
                unknown.append(m)

        if unknown:
            reason = f"unknown:{','.join(unknown[:5])}"
            audit['parse_status'] = 'failed'
            audit['exclusion_reason'] = reason
            audit['anchor_status'] = 'not_resolvable'
            audit['backbone_length'] = len(backbone)
            parse_failure_reasons['unknown_monomer'] += 1
            excluded_rows.append(audit)
            audit_rows.append(audit)
            continue

        if len(backbone) < 3:
            audit['parse_status'] = 'failed'
            audit['exclusion_reason'] = f"backbone_too_short:{len(backbone)}"
            audit['anchor_status'] = 'not_resolvable'
            audit['backbone_length'] = len(backbone)
            parse_failure_reasons[f'len_{len(backbone)}'] += 1
            excluded_rows.append(audit)
            audit_rows.append(audit)
            continue

        # —— Success ——
        sequence = ''.join(backbone)
        audit['parse_status'] = 'success'
        audit['exclusion_reason'] = ''
        audit['backbone_length'] = len(backbone)

        # Parse edits
        edits = conv.parse_edits(row_dict, sequence, sample_id)
        audit['n_edits'] = len(edits)

        # Anchor status
        if not edits:
            audit['anchor_status'] = 'no_edits'
        else:
            kinds = {e.anchor_kind for e in edits}
            if kinds == {'explicit'}:
                audit['anchor_status'] = 'explicit_anchor'
            elif 'explicit' in kinds:
                audit['anchor_status'] = 'mixed_explicit_inferred'
            elif 'inferred' in kinds:
                audit['anchor_status'] = 'weakly_inferable'
            else:
                audit['anchor_status'] = 'ambiguous'
        anchor_status_counts[audit['anchor_status']] += 1

        # Edit families & types (already strings from pydantic serialization)
        families_list = [e.edit_family for e in edits]
        types_list = [e.edit_type for e in edits]
        audit['edit_families'] = ';'.join(sorted(set(families_list))) if families_list else 'none'
        audit['edit_types'] = ';'.join(sorted(set(types_list))[:10]) if types_list else 'none'
        for f in set(families_list):
            edit_family_inventory[f] += 1
        for t in set(types_list):
            edit_type_inventory[t] += 1

        audit['n_terminal_edits'] = sum(1 for e in edits if e.edit_family == 'n_terminal')
        audit['c_terminal_edits'] = sum(1 for e in edits if e.edit_family == 'c_terminal')
        audit['cyclization'] = int(any(e.edit_family == 'cyclization' for e in edits))

        audit_rows.append(audit)

    # ═══ Step 3: Write outputs ════════════════════════════════════════════
    logger.info("Step 3/4: Writing per-sample audit CSV...")
    fieldnames = [
        'audit_index', 'csv_row', 'csv_id', 'sample_id', 'source', 'year',
        'parse_status', 'exclusion_reason', 'backbone_length',
        'anchor_status', 'n_edits', 'edit_families', 'edit_types',
        'n_terminal_edits', 'c_terminal_edits', 'cyclization',
        'molecule_shape', 'monomer_length', 'quality_flags',
    ]
    with open(output_dir / 'per_sample_audit.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(audit_rows)

    with open(output_dir / 'excluded_samples.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(excluded_rows)

    # ═══ Step 4: Monomer coverage & summary ═══════════════════════════════
    logger.info("Step 4/4: Monomer coverage & summary...")
    with open(output_dir / 'monomer_coverage.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['monomer', 'total_occurrences', 'mapped', 'mapped_to', 'in_failures', 'failure_count'])
        for monomer, total in monomer_counts.most_common():
            mapped = monomer in monomer_mapped
            mapped_to = conv.monomer_to_aa.get(monomer, '')
            w.writerow([monomer, total, mapped, mapped_to,
                       monomer in monomer_unmapped, monomer_unmapped.get(monomer, 0)])

    n_success = sum(1 for r in audit_rows if r['parse_status'] == 'success')
    n_failed = sum(1 for r in audit_rows if r['parse_status'] == 'failed')
    n_explicit = sum(1 for r in audit_rows if r.get('anchor_status') == 'explicit_anchor')
    n_cyclic = sum(1 for r in audit_rows if r.get('cyclization') == 1)

    summary = {
        'total_input_rows': len(df),
        'parse_success': n_success, 'parse_pct': round(100*n_success/len(df),2),
        'parse_failed': n_failed, 'fail_pct': round(100*n_failed/len(df),2),
        'unique_monomers': len(monomer_counts),
        'mapped_monomers': len(monomer_mapped),
        'unmapped_monomers': len(monomer_unmapped),
        'explicit_anchors': n_explicit, 'explicit_pct': round(100*n_explicit/max(n_success,1),2),
        'cyclic_samples': n_cyclic, 'cyclic_pct': round(100*n_cyclic/max(n_success,1),2),
        'parse_failure_reasons': dict(parse_failure_reasons),
        'anchor_status_dist': dict(anchor_status_counts),
        'quality_flags': dict(quality_flag_counts),
    }
    with open(output_dir / 'audit_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / 'audit_summary.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric','value'])
        for k, v in summary.items():
            if not isinstance(v, dict): w.writerow([k, v])

    # Methods
    methods = f"""# Dataset Audit — Methods

## Automated Audit Protocol
All {len(df):,} input rows were processed programmatically without manual sampling.

1. Monomer Census: {len(monomer_counts):,} unique monomers, {len(monomer_mapped):,} mapped ({len(monomer_unmapped):,} unmapped after v2.0)
2. Per-Sample Parsing: {n_success:,} success ({100*n_success/len(df):.1f}%), {n_failed:,} failed ({100*n_failed/len(df):.1f}%)
3. Anchor Classification: {n_explicit:,}/{n_success:,} explicit anchors ({100*n_explicit/max(n_success,1):.1f}%)
4. Quality Flags: {dict(quality_flag_counts)}

## Exclusion Transparency
All {n_failed:,} excluded rows with reasons in `excluded_samples.csv`.
No rows were silently dropped.
"""
    with open(output_dir / 'audit_methods_text.md', 'w') as f:
        f.write(methods)

    logger.info(f"DONE. Output: {output_dir}")
    logger.info(f"  per_sample_audit.csv: {len(audit_rows):,} rows")
    logger.info(f"  excluded_samples.csv: {len(excluded_rows):,} rows")
    logger.info(f"  monomer_coverage.csv: {len(monomer_counts):,} monomers")


if __name__ == '__main__':
    audit_all()
