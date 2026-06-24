#!/usr/bin/env python3
"""
Convert raw datasets to PEM unified schema.

Converts each dataset from raw CSV format to standardized PEM schema
with JSONL and Parquet serialization.

Usage:
    python scripts/convert_to_pem_schema.py --dataset cycpeptmpdb

Output:
    - data/processed/pem_schema/<dataset>.jsonl
    - data/processed/pem_schema/<dataset>.parquet
    - reports/schema_validation/<dataset>_conversion_report.json
    - reports/schema_validation/<dataset>_validation_report.md
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.converters import (
    CycPeptMPDBConverter,
)
from src.data.serialization import save_both_formats
from src.data.pem_schema import SchemaValidationReport


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_validation_report_md(
    converter,
    samples,
    output_path: Path
):
    """Generate markdown validation report."""

    report_lines = []

    report_lines.append(f"# Schema Validation Report: {converter.dataset_name}")
    report_lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n**Parser Version**: {converter.parser_version}")

    # Conversion statistics
    stats = converter.conversion_stats

    report_lines.append("\n## Conversion Statistics\n")
    report_lines.append("| Metric | Count | Percentage |")
    report_lines.append("|--------|-------|------------|")
    report_lines.append(f"| Total rows | {stats['total_rows']:,} | 100% |")
    report_lines.append(
        f"| Successfully converted | {stats['successful']:,} | "
        f"{100 * stats['successful'] / max(stats['total_rows'], 1):.1f}% |"
    )
    report_lines.append(
        f"| Failed | {stats['failed']:,} | "
        f"{100 * stats['failed'] / max(stats['total_rows'], 1):.1f}% |"
    )

    # Sample statistics
    report_lines.append("\n## Sample-Level Validation\n")
    report_lines.append(f"- **Valid samples**: {len(samples):,}")

    if samples:
        # Anchor status distribution
        from collections import Counter
        anchor_counts = Counter(s.anchor_status for s in samples)

        report_lines.append("\n### Anchor Status Distribution\n")
        report_lines.append("| Status | Count | Percentage |")
        report_lines.append("|--------|-------|------------|")

        for status, count in anchor_counts.most_common():
            pct = 100 * count / len(samples)
            report_lines.append(f"| {status} | {count:,} | {pct:.1f}% |")

        # Edit statistics
        total_edits = sum(len(s.edits) for s in samples)
        samples_with_edits = sum(1 for s in samples if s.edits)

        report_lines.append("\n### Edit Statistics\n")
        report_lines.append(f"- **Total edits**: {total_edits:,}")
        report_lines.append(f"- **Samples with edits**: {samples_with_edits:,}")
        report_lines.append(f"- **Samples without edits**: {len(samples) - samples_with_edits:,}")

        if total_edits > 0:
            avg_edits = total_edits / samples_with_edits if samples_with_edits > 0 else 0
            report_lines.append(f"- **Average edits per modified sample**: {avg_edits:.2f}")

            # Edit family distribution
            edit_families = Counter()
            edit_types = Counter()
            for sample in samples:
                for edit in sample.edits:
                    edit_families[edit.edit_family] += 1
                    edit_types[edit.edit_type] += 1

            report_lines.append("\n### Edit Family Distribution\n")
            report_lines.append("| Family | Count |")
            report_lines.append("|--------|-------|")
            for family, count in edit_families.most_common():
                report_lines.append(f"| {family} | {count:,} |")

            report_lines.append("\n### Top Edit Types\n")
            report_lines.append("| Type | Count |")
            report_lines.append("|------|-------|")
            for edit_type, count in edit_types.most_common(10):
                report_lines.append(f"| {edit_type} | {count:,} |")

    # Conversion errors
    if converter.conversion_errors:
        report_lines.append("\n## Conversion Errors\n")
        report_lines.append(f"\nTotal errors: {len(converter.conversion_errors)}\n")

        # Group by error type
        from collections import defaultdict
        errors_by_type = defaultdict(list)
        for error in converter.conversion_errors:
            errors_by_type[error['error_type']].append(error)

        report_lines.append("| Error Type | Count |")
        report_lines.append("|------------|-------|")
        for error_type, errors in sorted(errors_by_type.items(), key=lambda x: len(x[1]), reverse=True):
            report_lines.append(f"| {error_type} | {len(errors):,} |")

        # Show examples
        report_lines.append("\n### Example Errors\n")
        for error_type, errors in list(errors_by_type.items())[:3]:
            report_lines.append(f"\n**{error_type}** (showing first 3):\n")
            for error in errors[:3]:
                report_lines.append(f"- Sample `{error['sample_id']}`: {error['error']}")

    # Schema compliance
    report_lines.append("\n## Schema Compliance\n")

    if samples:
        report_lines.append("✓ All samples pass Pydantic validation")
        report_lines.append("\n### Schema Requirements Met\n")
        report_lines.append("- [x] Required fields present")
        report_lines.append("- [x] Type validation passed")
        report_lines.append("- [x] Anchor-position consistency validated")
        report_lines.append("- [x] Edit-residue matching validated")
        report_lines.append("- [x] Provenance tracking included")
    else:
        report_lines.append("⚠️ No valid samples to validate")

    # Recommendations
    report_lines.append("\n## Recommendations\n")

    success_rate = stats['successful'] / max(stats['total_rows'], 1)

    if success_rate >= 0.9:
        report_lines.append("✓ **High success rate** - Schema conversion successful")
    elif success_rate >= 0.7:
        report_lines.append("⚠️ **Moderate success rate** - Review conversion errors")
    else:
        report_lines.append("❌ **Low success rate** - Significant issues with conversion")

    if samples:
        no_edit_count = sum(1 for s in samples if not s.edits)
        if no_edit_count > len(samples) * 0.5:
            report_lines.append("\n⚠️ **Warning**: >50% of samples have no edits")
            report_lines.append("- Review modification parsing logic")
            report_lines.append("- Check source data format")

    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Validation report saved to {output_path}")


def convert_dataset(
    dataset_key: str,
    raw_data_dir: Path,
    output_dir: Path,
    report_dir: Path,
    strict_mode: bool = True
):
    """
    Convert a single dataset to PEM schema.

    Args:
        dataset_key: Dataset identifier
        raw_data_dir: Directory with raw CSV files
        output_dir: Output directory for PEM schema files
        report_dir: Directory for validation reports
        strict_mode: Use strict parsing mode
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Converting: {dataset_key}")
    logger.info(f"{'='*70}\n")

    # Map dataset key to file and converter
    dataset_config = {
        'cycpeptmpdb': {
            'file': raw_data_dir / 'cycpeptmpdb_pampa.csv',
            'converter_class': CycPeptMPDBConverter,
            'name': 'cycpeptmpdb_pampa'
        },
    }

    if dataset_key not in dataset_config:
        logger.error(f"Unknown dataset: {dataset_key}")
        return

    config = dataset_config[dataset_key]
    raw_file = config['file']

    if not raw_file.exists():
        logger.error(f"File not found: {raw_file}")
        logger.error(f"Please ensure data is available before conversion")
        return

    # Load raw data
    logger.info(f"Loading raw data from {raw_file}")
    df = pd.read_csv(raw_file)
    logger.info(f"Loaded {len(df)} rows")

    # Create converter
    converter = config['converter_class'](strict_mode=strict_mode)

    # Convert
    logger.info("Converting to PEM schema...")
    samples = converter.convert_dataframe(df, raw_file.name)

    logger.info(f"Conversion complete:")
    logger.info(f"  Total rows: {converter.conversion_stats['total_rows']}")
    logger.info(f"  Successful: {converter.conversion_stats['successful']}")
    logger.info(f"  Failed: {converter.conversion_stats['failed']}")

    if not samples:
        logger.warning("No samples successfully converted!")
        return

    # Save in both formats
    logger.info("Saving to JSONL and Parquet...")
    save_result = save_both_formats(samples, output_dir, config['name'])

    logger.info(f"Saved {save_result['jsonl']} samples to {save_result['jsonl_path']}")
    logger.info(f"Saved {save_result['parquet']} samples to {save_result['parquet_path']}")

    # Save conversion report
    conversion_report_path = report_dir / f"{config['name']}_conversion_report.json"
    converter.save_conversion_report(conversion_report_path)
    logger.info(f"Conversion report: {conversion_report_path}")

    # Generate validation report
    validation_report_path = report_dir / f"{config['name']}_validation_report.md"
    generate_validation_report_md(converter, samples, validation_report_path)
    logger.info(f"Validation report: {validation_report_path}")


def main():
    """Main conversion execution."""
    parser = argparse.ArgumentParser(
        description="Convert datasets to PEM unified schema"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cycpeptmpdb",
        choices=["cycpeptmpdb"],
        help="Dataset to convert (currently only cycpeptmpdb is supported)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Use strict parsing mode (default)"
    )
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="Use relaxed parsing mode"
    )

    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Must specify --dataset or --all")

    strict_mode = not args.relaxed

    # Setup paths
    raw_data_dir = PROJECT_ROOT / "data" / "raw"
    output_dir = PROJECT_ROOT / "data" / "processed" / "pem_schema"
    report_dir = PROJECT_ROOT / "reports" / "schema_validation"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Determine datasets to process
    datasets = [args.dataset]

    # Convert each dataset
    for dataset_key in datasets:
        try:
            convert_dataset(
                dataset_key,
                raw_data_dir,
                output_dir,
                report_dir,
                strict_mode=strict_mode
            )
        except Exception as e:
            logger.error(f"Error converting {dataset_key}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "="*70)
    logger.info("Conversion complete!")
    logger.info("="*70)
    logger.info(f"\nPEM schema files: {output_dir}")
    logger.info(f"Validation reports: {report_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Review validation reports in reports/schema_validation/")
    logger.info("2. Check conversion success rates")
    logger.info("3. Refine parsers if needed based on errors")
    logger.info("4. Proceed to Stage 3 (Edit Pair Identification)")


if __name__ == "__main__":
    main()
