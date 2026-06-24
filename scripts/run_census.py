#!/usr/bin/env python3
"""
Run census on all available datasets.

This script performs initial data auditing without any filtering or processing.
All findings are logged and reported for transparency.

Usage:
    python scripts/run_census.py [--datasets DATASET1 DATASET2 ...]

Examples:
    python scripts/run_census.py  # Run on all available datasets
    python scripts/run_census.py --datasets cycpeptmpdb pepmsnsd
"""

import sys
import argparse
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.census import DatasetCensus
from src.utils.logging import create_audit_logger


def main():
    """Run census on available datasets."""
    parser = argparse.ArgumentParser(
        description="Run census on PEM datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["cycpeptmpdb"],
        default=["all"],
        help="Datasets to census (default: all available)"
    )
    args = parser.parse_args()

    # Setup paths
    raw_data_dir = PROJECT_ROOT / "data" / "raw"
    census_output_dir = PROJECT_ROOT / "reports" / "01_census"
    census_output_dir.mkdir(parents=True, exist_ok=True)

    # Create audit logger
    logger = create_audit_logger(
        log_dir=census_output_dir,
        stage="census"
    )

    logger.log_info("="*60)
    logger.log_info("PEM Dataset Census")
    logger.log_info("="*60)

    # Define dataset configuration — single dataset for this study
    dataset_configs = {
        "cycpeptmpdb": {
            "file": raw_data_dir / "cycpeptmpdb_pampa.csv",
            "name": "CycPeptMPDB_PAMPA",
            "sequence_column": None,
            "property_column": None
        },
    }

    datasets_to_process = args.datasets

    # Create census tool
    census = DatasetCensus(logger)

    # Process each dataset
    reports = {}
    for dataset_key in datasets_to_process:
        config = dataset_configs[dataset_key]
        file_path = config["file"]

        logger.log_info("")
        logger.log_info("="*60)
        logger.log_info(f"Processing: {config['name']}")
        logger.log_info("="*60)

        if not file_path.exists():
            logger.log_warning(
                "missing_dataset",
                f"Dataset file not found: {file_path}",
                example={"dataset": dataset_key, "expected_path": str(file_path)}
            )
            print(f"\n⚠️  {config['name']} not found at {file_path}")
            print("Please download this dataset first.")
            continue

        try:
            # Run census
            report = census.census_csv(
                file_path=file_path,
                dataset_name=config["name"],
                sequence_column=config["sequence_column"],
                property_column=config["property_column"]
            )

            # Save report
            census.save_report(report, census_output_dir)

            reports[dataset_key] = report

            print(f"\n✓ Census complete for {config['name']}")
            print(f"  Records: {report.total_records}")
            print(f"  Report: {census_output_dir / config['name']}_census.md")

        except Exception as e:
            logger.log_warning(
                "census_error",
                f"Error processing {config['name']}: {str(e)}"
            )
            print(f"\n✗ Error processing {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary report
    logger.log_info("")
    logger.log_info("="*60)
    logger.log_info("Census Summary")
    logger.log_info("="*60)

    total_records = sum(r.total_records for r in reports.values())
    logger.log_statistic("total_records_across_datasets", total_records)

    print("\n" + "="*60)
    print("Census Summary")
    print("="*60)
    print(f"\nDatasets processed: {len(reports)}")
    print(f"Total records: {total_records}")
    print(f"\nReports saved to: {census_output_dir}")

    # Finalize audit log
    audit_report = logger.finalize()

    print(f"\nAudit log: {census_output_dir}")
    print("\nNext steps:")
    print("1. Review census reports in reports/01_census/")
    print("2. Update config/parsing_rules.yaml based on findings")
    print("3. Document any assumptions in PROJECT_LOG.md")
    print("4. Run: python scripts/convert_to_pem_schema.py --dataset cycpeptmpdb")


if __name__ == "__main__":
    main()
