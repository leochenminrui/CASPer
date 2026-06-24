#!/usr/bin/env python3
"""
Generate attachment-aware chemical representations for PEM edits.

Processes PEM schema files and generates TCS and AAF representations
for supported edit families.

Usage:
    python scripts/generate_chem_repr.py --dataset cycpeptmpdb
    python scripts/generate_chem_repr.py --all

Output:
    - data/processed/chem_repr/<dataset>_chem_repr.jsonl
    - reports/chem_repr/<dataset>_coverage_report.md
    - reports/chem_repr/<dataset>_failure_cases.md
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.serialization import load_jsonl
from src.data.chem_repr import canonicalize_edit, ParserStatus, ExclusionReason

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChemReprProcessor:
    """Process PEM samples to generate chemical representations."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

        # Statistics
        self.stats = {
            'total_samples': 0,
            'total_edits': 0,
            'covered_edits': 0,
            'excluded_edits': 0,
            'success': 0,
            'inferred': 0,
            'failed': 0
        }

        # Tracking
        self.covered_by_family = Counter()
        self.excluded_by_reason = Counter()
        self.success_examples = defaultdict(list)
        self.failure_examples = defaultdict(list)

        # Outputs
        self.chem_reprs = []

    def process_samples(self, samples):
        """Process PEM samples to generate chemical representations."""
        self.stats['total_samples'] = len(samples)

        for sample in samples:
            self.stats['total_edits'] += len(sample.edits)

            for edit in sample.edits:
                # Convert edit to dict format
                edit_data = {
                    'edit_id': edit.edit_id,
                    'edit_family': edit.edit_family,
                    'edit_type': edit.edit_type,
                    'anchor_positions': edit.anchor_positions,
                    'anchor_residues': edit.anchor_residues,
                    'chem_rep_raw': edit.chem_rep_raw
                }

                # Try to canonicalize
                chem_repr, error, notes = canonicalize_edit(
                    edit_data,
                    sample.sequence
                )

                if chem_repr:
                    # Success
                    self.stats['covered_edits'] += 1
                    self.covered_by_family[edit.edit_family] += 1

                    if chem_repr.parser_status == ParserStatus.SUCCESS:
                        self.stats['success'] += 1
                    elif chem_repr.parser_status == ParserStatus.INFERRED:
                        self.stats['inferred'] += 1

                    # Save representation
                    chem_repr_dict = chem_repr.to_dict()
                    chem_repr_dict['sample_id'] = sample.sample_id
                    chem_repr_dict['sequence'] = sample.sequence
                    self.chem_reprs.append(chem_repr_dict)

                    # Save example
                    key = f"{edit.edit_family}_{edit.edit_type}"
                    if len(self.success_examples[key]) < 3:
                        self.success_examples[key].append({
                            'sample_id': sample.sample_id,
                            'sequence': sample.sequence,
                            'tcs': chem_repr.to_tcs(),
                            'rule_id': chem_repr.rule_id,
                            'input_raw': edit.chem_rep_raw
                        })

                else:
                    # Failure
                    self.stats['excluded_edits'] += 1
                    self.stats['failed'] += 1

                    # Determine exclusion reason
                    if "unsupported" in notes.lower():
                        reason = "unsupported_modification"
                    elif "invalid" in notes.lower():
                        reason = "invalid_position"
                    else:
                        reason = "other"

                    self.excluded_by_reason[reason] += 1

                    # Save example
                    if len(self.failure_examples[reason]) < 3:
                        self.failure_examples[reason].append({
                            'sample_id': sample.sample_id,
                            'sequence': sample.sequence,
                            'edit_family': edit.edit_family,
                            'edit_type': edit.edit_type,
                            'error': error,
                            'notes': notes,
                            'input_raw': edit.chem_rep_raw
                        })

        logger.info(f"Processing complete:")
        logger.info(f"  Total edits: {self.stats['total_edits']}")
        logger.info(f"  Covered: {self.stats['covered_edits']}")
        logger.info(f"  Excluded: {self.stats['excluded_edits']}")

    def save_representations(self, output_path: Path):
        """Save chemical representations to JSONL."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for chem_repr in self.chem_reprs:
                f.write(json.dumps(chem_repr, default=str) + '\n')

        logger.info(f"Saved {len(self.chem_reprs)} representations to {output_path}")

    def generate_coverage_report(self, output_path: Path):
        """Generate coverage report markdown."""
        lines = []

        lines.append(f"# Chemical Representation Coverage Report")
        lines.append(f"\n**Dataset**: {self.dataset_name}")
        lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\n**Version**: 1.0.0")

        # Summary
        lines.append("\n## Summary\n")
        lines.append(f"- **Total samples**: {self.stats['total_samples']:,}")
        lines.append(f"- **Total edits**: {self.stats['total_edits']:,}")
        lines.append(f"- **Covered edits**: {self.stats['covered_edits']:,}")
        lines.append(f"- **Excluded edits**: {self.stats['excluded_edits']:,}")

        if self.stats['total_edits'] > 0:
            coverage_rate = 100 * self.stats['covered_edits'] / self.stats['total_edits']
            lines.append(f"- **Coverage rate**: {coverage_rate:.1f}%")

        # Coverage by family
        lines.append("\n## Coverage by Edit Family\n")
        lines.append("| Family | Count | % of Total Edits |")
        lines.append("|--------|-------|------------------|")

        for family, count in self.covered_by_family.most_common():
            pct = 100 * count / max(self.stats['total_edits'], 1)
            lines.append(f"| {family} | {count:,} | {pct:.1f}% |")

        # Parser status
        lines.append("\n## Parser Status Distribution\n")
        lines.append("| Status | Count | % of Covered |")
        lines.append("|--------|-------|--------------|")

        for status, key in [('Success', 'success'), ('Inferred', 'inferred')]:
            count = self.stats[key]
            pct = 100 * count / max(self.stats['covered_edits'], 1) if self.stats['covered_edits'] > 0 else 0
            lines.append(f"| {status} | {count:,} | {pct:.1f}% |")

        # Examples
        lines.append("\n## Success Examples\n")
        for key, examples in self.success_examples.items():
            lines.append(f"\n### {key}\n")
            for ex in examples[:3]:
                lines.append(f"\n**Sample**: `{ex['sample_id']}`")
                lines.append(f"- **TCS**: `{ex['tcs']}`")
                lines.append(f"- **Rule**: `{ex['rule_id']}`")
                lines.append(f"- **Input**: `{ex['input_raw']}`")

        # Save
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Coverage report saved to {output_path}")

    def generate_failure_report(self, output_path: Path):
        """Generate failure cases report."""
        lines = []

        lines.append(f"# Chemical Representation Failure Cases")
        lines.append(f"\n**Dataset**: {self.dataset_name}")
        lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Exclusions by reason
        lines.append("\n## Exclusions by Reason\n")
        lines.append("| Reason | Count | % of Total Edits |")
        lines.append("|--------|-------|------------------|")

        for reason, count in self.excluded_by_reason.most_common():
            pct = 100 * count / max(self.stats['total_edits'], 1)
            lines.append(f"| {reason} | {count:,} | {pct:.1f}% |")

        # Examples
        lines.append("\n## Failure Examples\n")
        for reason, examples in self.failure_examples.items():
            lines.append(f"\n### {reason}\n")
            for ex in examples[:3]:
                lines.append(f"\n**Sample**: `{ex['sample_id']}`")
                lines.append(f"- **Edit**: {ex['edit_family']} / {ex['edit_type']}")
                lines.append(f"- **Error**: {ex['error']}")
                lines.append(f"- **Notes**: {ex['notes']}")
                lines.append(f"- **Input**: `{ex['input_raw']}`")

        # Save
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Failure report saved to {output_path}")


def main():
    """Main processing."""
    parser = argparse.ArgumentParser(
        description="Generate chemical representations for PEM edits"
    )
    parser.add_argument(
        "--dataset",
        choices=["cycpeptmpdb"],
        help="Dataset to process"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all datasets"
    )

    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Must specify --dataset or --all")

    # Setup paths
    pem_schema_dir = PROJECT_ROOT / "data" / "processed" / "pem_schema"
    chem_repr_dir = PROJECT_ROOT / "data" / "processed" / "chem_repr"
    report_dir = PROJECT_ROOT / "reports" / "chem_repr"

    chem_repr_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Dataset mapping
    dataset_files = {
        'cycpeptmpdb': 'cycpeptmpdb_pampa.jsonl',
    }

    # Determine datasets to process
    datasets = [args.dataset]

    # Process each dataset
    for dataset_key in datasets:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {dataset_key}")
        logger.info(f"{'='*70}\n")

        pem_file = pem_schema_dir / dataset_files[dataset_key]

        if not pem_file.exists():
            logger.warning(f"PEM schema file not found: {pem_file}")
            logger.warning("Run Stage 2 conversion first")
            continue

        # Load PEM samples
        logger.info(f"Loading PEM samples from {pem_file}")
        samples = load_jsonl(pem_file)
        logger.info(f"Loaded {len(samples)} samples")

        # Process
        processor = ChemReprProcessor(dataset_key)
        processor.process_samples(samples)

        # Save outputs
        chem_repr_file = chem_repr_dir / f"{dataset_key}_chem_repr.jsonl"
        processor.save_representations(chem_repr_file)

        coverage_report = report_dir / f"{dataset_key}_coverage_report.md"
        processor.generate_coverage_report(coverage_report)

        failure_report = report_dir / f"{dataset_key}_failure_cases.md"
        processor.generate_failure_report(failure_report)

    logger.info("\n" + "="*70)
    logger.info("Chemical representation generation complete!")
    logger.info("="*70)
    logger.info(f"\nRepresentations: {chem_repr_dir}")
    logger.info(f"Reports: {report_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Review coverage reports")
    logger.info("2. Check failure cases")
    logger.info("3. Refine rules if needed")
    logger.info("4. Proceed to Stage 4 (Edit Pair Identification)")


if __name__ == "__main__":
    main()
