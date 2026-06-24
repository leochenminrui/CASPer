#!/usr/bin/env python3
"""
Preprocess CycPeptMPDB dataset to extract PAMPA-only samples.

This script:
1. Loads the raw CycPeptMPDB_Peptide_All.csv
2. Filters for samples with PAMPA measurements
3. Saves the filtered data for conversion
"""

import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "CycPeptMPDB" / "CycPeptMPDB_Peptide_All.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "cycpeptmpdb_pampa.csv"


def main():
    """Filter CycPeptMPDB for PAMPA-only samples."""

    logger.info("="*70)
    logger.info("CycPeptMPDB PAMPA Preprocessing")
    logger.info("="*70)

    # Load raw data
    logger.info(f"Loading raw data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    logger.info(f"Total samples: {len(df):,}")

    # Check PAMPA column
    logger.info(f"PAMPA non-null samples: {df['PAMPA'].notna().sum():,}")
    logger.info(f"PAMPA null samples: {df['PAMPA'].isna().sum():,}")

    # Filter for PAMPA only
    pampa_df = df[df['PAMPA'].notna()].copy()
    logger.info(f"Filtered to PAMPA-only: {len(pampa_df):,} samples")

    # Basic statistics
    logger.info("\nPAMPA value statistics:")
    logger.info(f"  Mean: {pampa_df['PAMPA'].mean():.2f}")
    logger.info(f"  Median: {pampa_df['PAMPA'].median():.2f}")
    logger.info(f"  Std: {pampa_df['PAMPA'].std():.2f}")
    logger.info(f"  Min: {pampa_df['PAMPA'].min():.2f}")
    logger.info(f"  Max: {pampa_df['PAMPA'].max():.2f}")

    # Check sequence availability
    seq_available = pampa_df['Sequence'].notna().sum()
    logger.info(f"\nSequence availability: {seq_available:,} / {len(pampa_df):,}")

    # Check molecule shape distribution
    logger.info("\nMolecule shape distribution:")
    shape_counts = pampa_df['Molecule_Shape'].value_counts()
    for shape, count in shape_counts.items():
        pct = 100 * count / len(pampa_df)
        logger.info(f"  {shape}: {count:,} ({pct:.1f}%)")

    # Save filtered data
    logger.info(f"\nSaving PAMPA-only data to: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pampa_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {len(pampa_df):,} samples")

    logger.info("\n" + "="*70)
    logger.info("Preprocessing complete!")
    logger.info("="*70)
    logger.info(f"\nNext step: Run conversion script")
    logger.info(f"  python scripts/convert_to_pem_schema.py --dataset cycpeptmpdb")


if __name__ == "__main__":
    main()
