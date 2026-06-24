#!/usr/bin/env python3
"""
Strict data census for PEM Stage 1.

Performs conservative auditing with explicit anchor classification
and detailed exclusion tracking.

Usage:
    python scripts/run_strict_census.py --dataset cycpeptmpdb
    python scripts/run_strict_census.py --all

Output:
    - reports/data_census/<dataset>.md
    - reports/data_census/<dataset>_metrics.json
    - data/exclusions/<dataset>_exclusions.csv
    - data/exclusions/<dataset>_all_records.csv
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audit import DatasetAuditor, AnchorResolvability, ExclusionReason, SampleAuditRecord
from src.data.census_report import CensusReportGenerator


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CycPeptMPDBAuditor:
    """
    Dataset-specific auditor for CycPeptMPDB (PAMPA subset).

    This will be refined once we see the actual data format.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def audit_dataset(self, raw_file: Path) -> DatasetAuditor:
        """
        Audit CycPeptMPDB PAMPA dataset.

        Args:
            raw_file: Path to raw CSV file

        Returns:
            DatasetAuditor with completed audit
        """
        logger.info(f"Starting CycPeptMPDB PAMPA audit (strict_mode={self.strict_mode})")

        auditor = DatasetAuditor("CycPeptMPDB_PAMPA", strict_mode=self.strict_mode)

        if not raw_file.exists():
            logger.error(f"File not found: {raw_file}")
            logger.error("Please download CycPeptMPDB (PAMPA subset) first.")
            logger.error("See DATA_ACQUISITION.md for instructions.")
            return auditor

        # Load data
        logger.info(f"Loading data from {raw_file}")
        df = pd.read_csv(raw_file)
        logger.info(f"Loaded {len(df)} raw samples")

        # Auto-detect columns (these are guesses - will refine with actual data)
        sequence_col = self._detect_column(df, ['sequence', 'seq', 'peptide', 'SMILES'])
        modification_col = self._detect_column(df, ['modification', 'mod', 'ptm', 'structure_details'])
        label_col = self._detect_column(df, ['permeability', 'logP', 'logPapp', 'Papp', 'pampa'])
        assay_col = self._detect_column(df, ['assay', 'assay_type', 'method'])

        logger.info(f"Detected columns:")
        logger.info(f"  Sequence: {sequence_col}")
        logger.info(f"  Modification: {modification_col}")
        logger.info(f"  Label: {label_col}")
        logger.info(f"  Assay: {assay_col}")

        if not sequence_col:
            logger.error("Cannot find sequence column - please check file format")
            return auditor

        # Filter for PAMPA assays only (strict separation)
        if assay_col:
            pampa_mask = df[assay_col].str.contains('PAMPA', case=False, na=False)
            df_pampa = df[pampa_mask].copy()
            logger.info(f"Filtered to PAMPA assays: {len(df_pampa)} samples")

            if len(df_pampa) < len(df):
                logger.warning(
                    f"Excluded {len(df) - len(df_pampa)} non-PAMPA samples "
                    "(strict assay separation)"
                )
        else:
            logger.warning("No assay column found - assuming all are PAMPA")
            df_pampa = df.copy()

        # Audit each sample
        for idx, row in df_pampa.iterrows():
            sample_id = f"CYCPEPT_{idx}"

            raw_data = {
                sequence_col: row.get(sequence_col) if sequence_col else None,
                modification_col: row.get(modification_col) if modification_col else None,
                label_col: row.get(label_col) if label_col else None,
                assay_col: row.get(assay_col) if assay_col else "PAMPA"
            }

            record = auditor.audit_sample(
                sample_id=sample_id,
                raw_data=raw_data,
                sequence_field=sequence_col,
                modification_field=modification_col,
                label_field=label_col,
                assay_field=assay_col
            )

            # CycPeptMPDB-specific anchor inference
            if record.has_sequence and record.has_modification:
                record.anchor_resolvability = self._infer_cycpept_anchor(
                    record.raw_sequence,
                    record.raw_modification
                )

                # Update exclusions based on anchor classification
                if record.anchor_resolvability == AnchorResolvability.NOT_RESOLVABLE:
                    if ExclusionReason.NO_ANCHOR not in record.exclusion_reasons:
                        record.add_exclusion(
                            ExclusionReason.NO_ANCHOR,
                            "Cannot resolve anchor for CycPeptMPDB format"
                        )
                elif (record.anchor_resolvability == AnchorResolvability.WEAKLY_INFERABLE
                      and self.strict_mode):
                    record.add_exclusion(
                        ExclusionReason.WEAK_ANCHOR_ONLY,
                        "Only weak anchor inference (strict mode)"
                    )

            auditor.audit_records.append(record)

        # Compute final metrics
        metrics = auditor.compute_metrics()

        logger.info(f"Audit complete:")
        logger.info(f"  Raw: {metrics.raw_sample_count}")
        logger.info(f"  Usable: {metrics.usable_total}")
        logger.info(f"  Explicit anchor: {metrics.explicit_anchor_count}")
        logger.info(f"  Weak anchor: {metrics.weakly_inferable_anchor_count}")

        return auditor

    def _detect_column(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        """Auto-detect column name from candidates."""
        for col in df.columns:
            col_lower = col.lower()
            if any(cand.lower() in col_lower for cand in candidates):
                return col
        return None

    def _infer_cycpept_anchor(self, sequence: str, modification: str) -> AnchorResolvability:
        """
        Infer anchor resolvability for CycPeptMPDB format.

        This is a placeholder - will be implemented based on actual format.

        Args:
            sequence: Sequence string
            modification: Modification string

        Returns:
            AnchorResolvability classification
        """
        # Placeholder logic - will refine with actual data
        if not sequence or not modification:
            return AnchorResolvability.NOT_RESOLVABLE

        # Example heuristics (to be refined):
        # - If modification contains position numbers → EXPLICIT
        # - If modification is single character different → WEAKLY_INFERABLE
        # - Otherwise → NOT_RESOLVABLE

        mod_str = str(modification).lower()

        # Check for explicit position markers
        if any(marker in mod_str for marker in ['pos', 'position', '[', ']', '@']):
            return AnchorResolvability.EXPLICIT_ANCHOR

        # Check if it's a simple modification that might be inferable
        if len(mod_str) < 20:  # Short modification string
            return AnchorResolvability.WEAKLY_INFERABLE

        return AnchorResolvability.NOT_RESOLVABLE


# PepMSNDAuditor and DBAASPAuditor classes were removed before submission.
# See archived/future_extension/ for the original skeleton code.

def main():
    """Main census execution."""
    parser = argparse.ArgumentParser(
        description="Strict data census for PEM Stage 1"
    )
    parser.add_argument(
        "--dataset",
        choices=["cycpeptmpdb"],
        help="Dataset to audit"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Audit all available datasets"
    )
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="Use relaxed mode (allow weak anchors)"
    )

    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Must specify --dataset or --all")

    strict_mode = not args.relaxed

    # Setup paths
    raw_data_dir = PROJECT_ROOT / "data" / "raw"
    census_output_dir = PROJECT_ROOT / "reports" / "data_census"
    exclusion_dir = PROJECT_ROOT / "data" / "exclusions"

    census_output_dir.mkdir(parents=True, exist_ok=True)
    exclusion_dir.mkdir(parents=True, exist_ok=True)

    # Create report generator
    report_gen = CensusReportGenerator(census_output_dir)

    # Datasets to process — single dataset for this study
    datasets_to_audit = [
        ("cycpeptmpdb", CycPeptMPDBAuditor),
    ]

    # Run audits
    all_metrics = {}

    for dataset_key, auditor_class in datasets_to_audit:
        logger.info(f"\n{'='*70}")
        logger.info(f"Auditing: {dataset_key}")
        logger.info(f"{'='*70}\n")

        # Map dataset key to file — single dataset for this study
        file_mapping = {
            "cycpeptmpdb": raw_data_dir / "cycpeptmpdb_pampa.csv",
        }

        raw_file = file_mapping[dataset_key]

        # Create auditor and run
        auditor_instance = auditor_class(strict_mode=strict_mode)
        auditor = auditor_instance.audit_dataset(raw_file)

        if auditor.metrics.raw_sample_count == 0:
            logger.warning(f"No samples found for {dataset_key} - skipping")
            continue

        # Save exclusion records
        exclusion_file = exclusion_dir / f"{dataset_key}_exclusions.csv"
        n_excluded = auditor.save_exclusion_csv(str(exclusion_file))
        logger.info(f"Saved {n_excluded} exclusion records to {exclusion_file}")

        # Save all records
        all_records_file = exclusion_dir / f"{dataset_key}_all_records.csv"
        n_all = auditor.save_all_records_csv(str(all_records_file))
        logger.info(f"Saved {n_all} audit records to {all_records_file}")

        # Generate report
        report_path = report_gen.generate_dataset_report(
            auditor.metrics,
            dataset_description=f"Strict census for {dataset_key}",
            additional_notes=[
                f"Mode: {'Strict' if strict_mode else 'Relaxed'}",
                "All anchor classifications are explicitly labeled",
                "No optimistic assumptions made"
            ]
        )

        logger.info(f"Generated report: {report_path}")

        all_metrics[auditor.metrics.dataset_name] = auditor.metrics

    # Generate combined summary
    if len(all_metrics) > 1:
        logger.info(f"\n{'='*70}")
        logger.info("Generating combined summary")
        logger.info(f"{'='*70}\n")

        summary_path = report_gen.generate_combined_summary(all_metrics)
        logger.info(f"Generated combined summary: {summary_path}")

    logger.info("\n" + "="*70)
    logger.info("Census complete!")
    logger.info("="*70)
    logger.info(f"\nReports saved to: {census_output_dir}")
    logger.info(f"Exclusion logs saved to: {exclusion_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Review census reports in reports/data_census/")
    logger.info("2. Check exclusion CSVs in data/exclusions/")
    logger.info("3. Update PROJECT_LOG.md with findings")
    logger.info("4. Refine anchor inference methods based on actual formats")
    logger.info("5. Proceed to Stage 2 (Processing) if datasets are viable")


if __name__ == "__main__":
    main()
