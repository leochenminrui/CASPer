"""
Dataset census tools for initial data auditing.

This module provides tools to systematically analyze raw datasets before processing,
documenting:
- Size and format
- Field distributions
- Missing data patterns
- Modification annotation schemes
- Property value distributions
- Potential edit pair candidates

All findings are logged and reported for transparency.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

from ..utils.logging import AuditLogger


@dataclass
class CensusReport:
    """Summary report from dataset census."""
    dataset_name: str
    file_path: str
    total_records: int
    fields: List[str]
    missing_data: Dict[str, int]
    property_stats: Dict[str, Any]
    sequence_length_stats: Dict[str, float]
    modification_patterns: Dict[str, Any]
    sample_records: List[Dict[str, Any]]
    data_quality_issues: List[str]
    edit_pair_potential: Dict[str, Any]


class DatasetCensus:
    """
    Conducts comprehensive census of a raw dataset.

    Analyzes data without making filtering decisions - just reports findings.
    """

    def __init__(self, logger: AuditLogger):
        """
        Initialize census tool.

        Args:
            logger: Audit logger for recording findings
        """
        self.logger = logger

    def census_csv(
        self,
        file_path: Path,
        dataset_name: str,
        sequence_column: Optional[str] = None,
        property_column: Optional[str] = None
    ) -> CensusReport:
        """
        Perform census on CSV file.

        Args:
            file_path: Path to CSV file
            dataset_name: Name of dataset
            sequence_column: Name of sequence column (auto-detect if None)
            property_column: Name of property column (auto-detect if None)

        Returns:
            CensusReport with findings
        """
        self.logger.log_info(f"Starting census for {dataset_name}")
        self.logger.log_info(f"File: {file_path}")

        # Load data
        df = pd.read_csv(file_path)
        total_records = len(df)

        self.logger.log_statistic(f"{dataset_name}_total_records", total_records)
        self.logger.log_info(f"Loaded {total_records} records")

        # Analyze fields
        fields = df.columns.tolist()
        self.logger.log_info(f"Fields: {fields}")

        # Auto-detect sequence and property columns if not provided
        if sequence_column is None:
            sequence_column = self._detect_sequence_column(df)
            self.logger.log_info(f"Auto-detected sequence column: {sequence_column}")

        if property_column is None:
            property_column = self._detect_property_column(df, dataset_name)
            self.logger.log_info(f"Auto-detected property column: {property_column}")

        # Analyze missing data
        missing_data = self._analyze_missing_data(df)

        # Analyze property distribution
        property_stats = self._analyze_property_distribution(
            df, property_column
        ) if property_column else {}

        # Analyze sequence lengths
        sequence_length_stats = self._analyze_sequence_lengths(
            df, sequence_column
        ) if sequence_column else {}

        # Analyze modification patterns
        modification_patterns = self._analyze_modifications(
            df, sequence_column
        ) if sequence_column else {}

        # Sample records for inspection
        sample_records = self._get_sample_records(df, n_samples=5)

        # Identify data quality issues
        quality_issues = self._identify_quality_issues(
            df, sequence_column, property_column
        )

        # Assess edit pair potential
        edit_pair_potential = self._assess_edit_pairs(
            df, sequence_column
        ) if sequence_column else {}

        report = CensusReport(
            dataset_name=dataset_name,
            file_path=str(file_path),
            total_records=total_records,
            fields=fields,
            missing_data=missing_data,
            property_stats=property_stats,
            sequence_length_stats=sequence_length_stats,
            modification_patterns=modification_patterns,
            sample_records=sample_records,
            data_quality_issues=quality_issues,
            edit_pair_potential=edit_pair_potential
        )

        self.logger.log_info(f"Census complete for {dataset_name}")
        return report

    def _detect_sequence_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect sequence column by name patterns."""
        candidates = ['sequence', 'seq', 'peptide', 'protein', 'smiles']
        for col in df.columns:
            if any(c in col.lower() for c in candidates):
                return col
        return None

    def _detect_property_column(
        self,
        df: pd.DataFrame,
        dataset_name: str
    ) -> Optional[str]:
        """Auto-detect property column based on dataset type."""
        # Dataset-specific heuristics
        if 'cycpept' in dataset_name.lower():
            candidates = ['permeability', 'logp', 'papp']
        else:
            candidates = ['value', 'target', 'label', 'activity']

        for col in df.columns:
            if any(c in col.lower() for c in candidates):
                return col
        return None

    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, int]:
        """Count missing values per column."""
        missing = df.isnull().sum().to_dict()
        missing = {k: int(v) for k, v in missing.items() if v > 0}

        for col, count in missing.items():
            pct = 100 * count / len(df)
            self.logger.log_warning(
                "missing_data",
                f"Column '{col}' has {count} missing values ({pct:.1f}%)"
            )

        return missing

    def _analyze_property_distribution(
        self,
        df: pd.DataFrame,
        property_column: str
    ) -> Dict[str, Any]:
        """Analyze distribution of target property."""
        values = df[property_column].dropna()

        if len(values) == 0:
            return {"error": "No valid values"}

        stats = {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75))
        }

        self.logger.log_info(f"Property statistics: {stats}")
        return stats

    def _analyze_sequence_lengths(
        self,
        df: pd.DataFrame,
        sequence_column: str
    ) -> Dict[str, float]:
        """Analyze peptide sequence lengths."""
        # This is a simple version - actual implementation depends on format
        lengths = df[sequence_column].dropna().apply(lambda x: len(str(x)))

        stats = {
            "mean": float(lengths.mean()),
            "std": float(lengths.std()),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "median": float(lengths.median())
        }

        self.logger.log_info(f"Sequence length statistics: {stats}")
        return stats

    def _analyze_modifications(
        self,
        df: pd.DataFrame,
        sequence_column: str
    ) -> Dict[str, Any]:
        """
        Analyze modification annotation patterns.

        This is dataset-specific and will need to be customized.
        """
        sequences = df[sequence_column].dropna()

        # Detect notation patterns
        patterns = {
            "contains_brackets": 0,  # e.g., [Ac]
            "contains_dashes": 0,     # e.g., Ac-
            "contains_lowercase": 0,  # e.g., dAla
            "contains_special_chars": 0,  # various
            "all_uppercase": 0        # standard sequences
        }

        for seq in sequences:
            seq_str = str(seq)
            if '[' in seq_str or ']' in seq_str:
                patterns["contains_brackets"] += 1
            if '-' in seq_str:
                patterns["contains_dashes"] += 1
            if any(c.islower() for c in seq_str):
                patterns["contains_lowercase"] += 1
            if any(c in seq_str for c in ['(', ')', '{', '}', '*', '#']):
                patterns["contains_special_chars"] += 1
            if seq_str.isupper() and seq_str.isalpha():
                patterns["all_uppercase"] += 1

        self.logger.log_info(f"Modification patterns: {patterns}")

        # Get examples of each pattern
        examples = {}
        for pattern in patterns:
            if patterns[pattern] > 0:
                # Find first example
                for seq in sequences:
                    seq_str = str(seq)
                    if self._matches_pattern(seq_str, pattern):
                        examples[pattern] = seq_str
                        break

        return {
            "patterns": patterns,
            "examples": examples
        }

    def _matches_pattern(self, seq: str, pattern: str) -> bool:
        """Check if sequence matches a pattern."""
        if pattern == "contains_brackets":
            return '[' in seq or ']' in seq
        elif pattern == "contains_dashes":
            return '-' in seq
        elif pattern == "contains_lowercase":
            return any(c.islower() for c in seq)
        elif pattern == "contains_special_chars":
            return any(c in seq for c in ['(', ')', '{', '}', '*', '#'])
        elif pattern == "all_uppercase":
            return seq.isupper() and seq.isalpha()
        return False

    def _get_sample_records(
        self,
        df: pd.DataFrame,
        n_samples: int = 5
    ) -> List[Dict[str, Any]]:
        """Get sample records for inspection."""
        sample = df.head(n_samples)
        return sample.to_dict('records')

    def _identify_quality_issues(
        self,
        df: pd.DataFrame,
        sequence_column: Optional[str],
        property_column: Optional[str]
    ) -> List[str]:
        """Identify potential data quality issues."""
        issues = []

        # Check for duplicates
        if sequence_column:
            n_duplicates = df[sequence_column].duplicated().sum()
            if n_duplicates > 0:
                issues.append(f"{n_duplicates} duplicate sequences found")
                self.logger.log_warning(
                    "duplicates",
                    f"Found {n_duplicates} duplicate sequences"
                )

        # Check for extreme values
        if property_column:
            values = df[property_column].dropna()
            if len(values) > 0:
                mean, std = values.mean(), values.std()
                outliers = ((values - mean).abs() > 3 * std).sum()
                if outliers > 0:
                    issues.append(f"{outliers} potential outliers (>3 std)")

        return issues

    def _assess_edit_pairs(
        self,
        df: pd.DataFrame,
        sequence_column: str
    ) -> Dict[str, Any]:
        """
        Assess potential for finding edit pairs.

        This is a rough estimate - actual pair finding happens later.
        """
        sequences = df[sequence_column].dropna().tolist()

        # Count sequences by length
        length_distribution = Counter(len(str(s)) for s in sequences)

        # Rough estimate: sequences with same length could be pairs
        potential_pairs = 0
        for length, count in length_distribution.items():
            if count > 1:
                potential_pairs += count * (count - 1) // 2

        return {
            "total_sequences": len(sequences),
            "unique_sequences": len(set(str(s) for s in sequences)),
            "length_distribution": dict(length_distribution),
            "rough_potential_pairs": potential_pairs,
            "note": "Actual edit pairs will be identified during processing"
        }

    def save_report(
        self,
        report: CensusReport,
        output_dir: Path
    ):
        """
        Save census report to file.

        Args:
            report: Census report to save
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        json_file = output_dir / f"{report.dataset_name}_census.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)

        self.logger.log_info(f"Census report saved to {json_file}")

        # Save markdown summary
        md_file = output_dir / f"{report.dataset_name}_census.md"
        self._save_markdown_report(report, md_file)

    def _save_markdown_report(
        self,
        report: CensusReport,
        output_file: Path
    ):
        """Generate human-readable markdown report."""
        with open(output_file, 'w') as f:
            f.write(f"# Census Report: {report.dataset_name}\n\n")
            f.write(f"**File**: `{report.file_path}`\n\n")
            f.write(f"**Total Records**: {report.total_records}\n\n")

            f.write("## Fields\n\n")
            for field in report.fields:
                f.write(f"- {field}\n")

            f.write("\n## Missing Data\n\n")
            if report.missing_data:
                for col, count in report.missing_data.items():
                    pct = 100 * count / report.total_records
                    f.write(f"- **{col}**: {count} ({pct:.1f}%)\n")
            else:
                f.write("No missing data detected.\n")

            if report.property_stats:
                f.write("\n## Property Statistics\n\n")
                f.write(f"```json\n{json.dumps(report.property_stats, indent=2)}\n```\n")

            if report.sequence_length_stats:
                f.write("\n## Sequence Length Statistics\n\n")
                f.write(f"```json\n{json.dumps(report.sequence_length_stats, indent=2)}\n```\n")

            if report.modification_patterns:
                f.write("\n## Modification Patterns\n\n")
                f.write(f"```json\n{json.dumps(report.modification_patterns, indent=2)}\n```\n")

            if report.data_quality_issues:
                f.write("\n## Data Quality Issues\n\n")
                for issue in report.data_quality_issues:
                    f.write(f"- {issue}\n")

            if report.edit_pair_potential:
                f.write("\n## Edit Pair Potential\n\n")
                f.write(f"```json\n{json.dumps(report.edit_pair_potential, indent=2)}\n```\n")

            f.write("\n## Sample Records\n\n")
            f.write(f"```json\n{json.dumps(report.sample_records, indent=2, default=str)}\n```\n")

        self.logger.log_info(f"Markdown report saved to {output_file}")
