"""
Audit logging utilities for PEM project.

Ensures all data exclusions, parsing decisions, and filtering steps are logged
with full transparency.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict


class AuditLogger:
    """
    Comprehensive audit logger for data processing decisions.

    Ensures no silent filtering - every exclusion is recorded with:
    - Count
    - Examples
    - Reason
    - Stage of pipeline
    """

    def __init__(self, log_dir: Path, stage: str):
        """
        Initialize audit logger.

        Args:
            log_dir: Directory to save audit logs
            stage: Pipeline stage (e.g., 'census', 'processing', 'training')
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.stage = stage
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Track exclusions
        self.exclusions = defaultdict(list)
        self.exclusion_counts = defaultdict(int)

        # Track warnings
        self.warnings = defaultdict(list)
        self.warning_counts = defaultdict(int)

        # Track statistics
        self.statistics = {}

        # Setup file logging
        log_file = self.log_dir / f"{stage}_{self.timestamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"PEM.{stage}")

        self.logger.info(f"=== Starting {stage} stage ===")
        self.logger.info(f"Timestamp: {self.timestamp}")

    def log_exclusion(
        self,
        reason: str,
        example: Optional[Dict[str, Any]] = None,
        details: Optional[str] = None
    ):
        """
        Log a data exclusion with reason and example.

        Args:
            reason: Exclusion reason (category)
            example: Example data point that was excluded
            details: Additional details about the exclusion
        """
        self.exclusion_counts[reason] += 1

        entry = {
            "count": self.exclusion_counts[reason],
            "timestamp": datetime.now().isoformat(),
        }

        if example is not None:
            entry["example"] = example

        if details is not None:
            entry["details"] = details

        # Only store first few examples to avoid memory issues
        if len(self.exclusions[reason]) < 10:
            self.exclusions[reason].append(entry)

        self.logger.warning(f"EXCLUSION [{reason}]: {details if details else 'See example'}")

    def log_warning(
        self,
        category: str,
        message: str,
        example: Optional[Dict[str, Any]] = None
    ):
        """
        Log a warning (non-fatal issue).

        Args:
            category: Warning category
            message: Warning message
            example: Example data point
        """
        self.warning_counts[category] += 1

        entry = {
            "count": self.warning_counts[category],
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }

        if example is not None:
            entry["example"] = example

        if len(self.warnings[category]) < 10:
            self.warnings[category].append(entry)

        self.logger.warning(f"WARNING [{category}]: {message}")

    def log_statistic(self, name: str, value: Any):
        """Log a summary statistic."""
        self.statistics[name] = value
        self.logger.info(f"STAT [{name}]: {value}")

    def log_info(self, message: str):
        """Log general informational message."""
        self.logger.info(message)

    def save_report(self, output_file: Optional[Path] = None):
        """
        Save comprehensive audit report.

        Args:
            output_file: Output file path (defaults to timestamped file)
        """
        if output_file is None:
            output_file = self.log_dir / f"{self.stage}_report_{self.timestamp}.json"

        report = {
            "stage": self.stage,
            "timestamp": self.timestamp,
            "statistics": self.statistics,
            "exclusions": {
                "summary": dict(self.exclusion_counts),
                "details": dict(self.exclusions)
            },
            "warnings": {
                "summary": dict(self.warning_counts),
                "details": dict(self.warnings)
            }
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Audit report saved to {output_file}")

        # Print summary
        self.logger.info("=== Exclusion Summary ===")
        for reason, count in self.exclusion_counts.items():
            self.logger.info(f"  {reason}: {count}")

        total_excluded = sum(self.exclusion_counts.values())
        self.logger.info(f"  TOTAL EXCLUDED: {total_excluded}")

        if self.warning_counts:
            self.logger.info("=== Warning Summary ===")
            for category, count in self.warning_counts.items():
                self.logger.info(f"  {category}: {count}")

        return report

    def finalize(self):
        """Finalize and save the audit log."""
        self.logger.info(f"=== Completing {self.stage} stage ===")
        return self.save_report()


def create_audit_logger(log_dir: Path, stage: str) -> AuditLogger:
    """
    Factory function to create an audit logger.

    Args:
        log_dir: Directory for audit logs
        stage: Pipeline stage name

    Returns:
        Configured AuditLogger instance
    """
    return AuditLogger(log_dir, stage)
