"""
Census report generation for PEM Stage 1.

Generates detailed markdown reports from census metrics.
"""

from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime

from .audit import CensusMetrics


class CensusReportGenerator:
    """Generates markdown census reports."""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_dataset_report(
        self,
        metrics: CensusMetrics,
        dataset_description: str = "",
        additional_notes: List[str] = None
    ) -> Path:
        """
        Generate detailed markdown report for a single dataset.

        Args:
            metrics: Census metrics
            dataset_description: Optional description
            additional_notes: Optional additional notes

        Returns:
            Path to generated report
        """
        report_lines = []

        # Header
        report_lines.append(f"# Data Census Report: {metrics.dataset_name}")
        report_lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"\n**Stage**: 1 - Dataset Census")

        if dataset_description:
            report_lines.append(f"\n## Dataset Description\n\n{dataset_description}")

        # Executive Summary
        report_lines.append("\n## Executive Summary\n")
        report_lines.append(f"- **Raw Samples**: {metrics.raw_sample_count:,}")
        report_lines.append(f"- **Parseable Samples**: {metrics.parseable_sample_count:,}")
        report_lines.append(f"- **Usable Samples (Conservative)**: {metrics.usable_total:,}")

        if metrics.raw_sample_count > 0:
            usable_pct = 100 * metrics.usable_total / metrics.raw_sample_count
            report_lines.append(f"- **Usable Rate**: {usable_pct:.1f}%")

        # Progressive Filtering Breakdown
        report_lines.append("\n## Progressive Filtering Breakdown\n")
        report_lines.append("| Stage | Count | % of Raw | Notes |")
        report_lines.append("|-------|-------|----------|-------|")

        def add_row(stage, count, notes=""):
            pct = 100 * count / max(metrics.raw_sample_count, 1)
            report_lines.append(f"| {stage} | {count:,} | {pct:.1f}% | {notes} |")

        add_row("Raw samples", metrics.raw_sample_count, "Total input")
        add_row("Parseable", metrics.parseable_sample_count, "Valid format")
        add_row("With sequence", metrics.sequence_available_count, "Sequence field present")
        add_row("With modification", metrics.modification_available_count, "Modification info present")
        add_row("With label", metrics.continuous_label_count, "Numeric label available")
        add_row("Explicit anchor", metrics.explicit_anchor_count, "Anchor directly annotated")
        add_row("Weak anchor", metrics.weakly_inferable_anchor_count, "Anchor inferable (flagged)")
        add_row("**Usable (conservative)**", metrics.usable_total, "Final usable count")

        # Anchor Resolvability Analysis
        report_lines.append("\n## Anchor Resolvability Classification\n")
        report_lines.append("**Classification Levels**:")
        report_lines.append("- **Explicit Anchor**: Direct annotation of edit position")
        report_lines.append("- **Weakly Inferable**: Can be inferred with heuristics (flagged as inferred)")
        report_lines.append("- **Not Resolvable**: Cannot determine anchor position\n")

        report_lines.append("| Classification | Count | % of Samples with Modifications |")
        report_lines.append("|----------------|-------|----------------------------------|")

        mod_count = max(metrics.modification_available_count, 1)
        report_lines.append(
            f"| Explicit Anchor | {metrics.explicit_anchor_count:,} | "
            f"{100 * metrics.explicit_anchor_count / mod_count:.1f}% |"
        )
        report_lines.append(
            f"| Weakly Inferable | {metrics.weakly_inferable_anchor_count:,} | "
            f"{100 * metrics.weakly_inferable_anchor_count / mod_count:.1f}% |"
        )
        report_lines.append(
            f"| Not Resolvable | {metrics.not_resolvable_anchor_count:,} | "
            f"{100 * metrics.not_resolvable_anchor_count / mod_count:.1f}% |"
        )

        # Data Quality Issues
        report_lines.append("\n## Data Quality Issues\n")
        report_lines.append("| Issue Type | Count | % of Raw |")
        report_lines.append("|------------|-------|----------|")

        def add_issue_row(issue_type, count):
            if count > 0:
                pct = 100 * count / max(metrics.raw_sample_count, 1)
                report_lines.append(f"| {issue_type} | {count:,} | {pct:.1f}% |")

        add_issue_row("Missing sequence", metrics.missing_sequence_count)
        add_issue_row("Unparseable format", metrics.unparseable_samples)
        add_issue_row("No modification info", metrics.no_modification_count)
        add_issue_row("Ambiguous modification", metrics.ambiguous_modification_count)
        add_issue_row("Missing label", metrics.missing_label_count)
        add_issue_row("Non-numeric label", metrics.non_numeric_label_count)
        add_issue_row("Duplicates", metrics.duplicate_count)
        add_issue_row("Multi-edit samples", metrics.multi_edit_count)

        # Exclusion Summary
        if metrics.exclusion_counts:
            report_lines.append("\n## Detailed Exclusion Breakdown\n")
            report_lines.append("| Exclusion Reason | Count |")
            report_lines.append("|------------------|-------|")

            for reason, count in sorted(
                metrics.exclusion_counts.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                report_lines.append(f"| {reason} | {count:,} |")

            total_exclusions = sum(metrics.exclusion_counts.values())
            report_lines.append(f"| **TOTAL EXCLUSIONS** | **{total_exclusions:,}** |")

        # Assay Type Distribution
        if metrics.assay_type_distribution:
            report_lines.append("\n## Assay Type Distribution\n")
            report_lines.append("| Assay Type | Count | % of Total |")
            report_lines.append("|------------|-------|------------|")

            for assay_type, count in sorted(
                metrics.assay_type_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                pct = 100 * count / max(metrics.raw_sample_count, 1)
                report_lines.append(f"| {assay_type} | {count:,} | {pct:.1f}% |")

        # Edit Type Analysis
        if metrics.single_edit_count > 0 or metrics.multi_edit_count > 0:
            report_lines.append("\n## Edit Type Analysis\n")
            report_lines.append("| Edit Type | Count |")
            report_lines.append("|-----------|-------|")
            report_lines.append(f"| Single edit | {metrics.single_edit_count:,} |")
            report_lines.append(f"| Multi-edit | {metrics.multi_edit_count:,} |")

        # Split Feasibility
        report_lines.append("\n## Split Feasibility Assessment\n")

        for note in metrics.split_feasibility_notes:
            report_lines.append(f"- {note}")

        # Recommendations
        report_lines.append("\n## Recommendations\n")

        if metrics.usable_total < 100:
            report_lines.append("⚠️ **CRITICAL**: Dataset has insufficient usable samples for modeling.")
            report_lines.append("\nActions:")
            report_lines.append("- Review exclusion criteria for potential relaxation")
            report_lines.append("- Consider combining with other datasets")
            report_lines.append("- Seek additional data sources")

        elif metrics.usable_total < 300:
            report_lines.append("⚠️ **WARNING**: Dataset is marginal for robust evaluation.")
            report_lines.append("\nActions:")
            report_lines.append("- Use cross-validation instead of fixed splits")
            report_lines.append("- Be cautious about overfitting")
            report_lines.append("- Document limited sample size in results")

        else:
            report_lines.append("✓ Dataset has sufficient samples for modeling.")

        if metrics.explicit_anchor_count == 0:
            report_lines.append("\n⚠️ **ANCHOR ISSUE**: No explicit anchors found.")
            report_lines.append("\nActions:")
            report_lines.append("- Review modification annotation format")
            report_lines.append("- Implement dataset-specific anchor inference")
            report_lines.append("- Document anchor inference method explicitly")

        # Additional Notes
        if additional_notes:
            report_lines.append("\n## Additional Notes\n")
            for note in additional_notes:
                report_lines.append(f"- {note}")

        # Save report
        report_path = self.output_dir / f"{metrics.dataset_name.lower().replace(' ', '_')}.md"
        report_text = "\n".join(report_lines)

        with open(report_path, 'w') as f:
            f.write(report_text)

        # Also save JSON metrics
        json_path = self.output_dir / f"{metrics.dataset_name.lower().replace(' ', '_')}_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2, default=str)

        return report_path

    def generate_combined_summary(
        self,
        all_metrics: Dict[str, CensusMetrics]
    ) -> Path:
        """
        Generate combined summary report across all datasets.

        Args:
            all_metrics: Dictionary of dataset_name -> CensusMetrics

        Returns:
            Path to combined summary report
        """
        report_lines = []

        # Header
        report_lines.append("# Combined Dataset Census Summary")
        report_lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"\n**Stage**: 1 - Dataset Census")
        report_lines.append(f"\n**Datasets Analyzed**: {len(all_metrics)}")

        # Cross-dataset comparison
        report_lines.append("\n## Cross-Dataset Comparison\n")
        report_lines.append("| Dataset | Raw | Parseable | Usable | Usable % | Explicit Anchor | Weak Anchor |")
        report_lines.append("|---------|-----|-----------|--------|----------|-----------------|-------------|")

        total_raw = 0
        total_usable = 0

        for dataset_name, metrics in all_metrics.items():
            raw = metrics.raw_sample_count
            parseable = metrics.parseable_sample_count
            usable = metrics.usable_total
            usable_pct = 100 * usable / max(raw, 1)
            explicit = metrics.explicit_anchor_count
            weak = metrics.weakly_inferable_anchor_count

            report_lines.append(
                f"| {dataset_name} | {raw:,} | {parseable:,} | {usable:,} | "
                f"{usable_pct:.1f}% | {explicit:,} | {weak:,} |"
            )

            total_raw += raw
            total_usable += usable

        report_lines.append(
            f"| **TOTAL** | **{total_raw:,}** | - | **{total_usable:,}** | "
            f"{100 * total_usable / max(total_raw, 1):.1f}% | - | - |"
        )

        # Anchor resolvability summary
        report_lines.append("\n## Anchor Resolvability Summary\n")
        report_lines.append("| Dataset | Explicit | Weakly Inferable | Not Resolvable | Total with Mods |")
        report_lines.append("|---------|----------|------------------|----------------|-----------------|")

        for dataset_name, metrics in all_metrics.items():
            explicit = metrics.explicit_anchor_count
            weak = metrics.weakly_inferable_anchor_count
            not_res = metrics.not_resolvable_anchor_count
            total_mods = metrics.modification_available_count

            report_lines.append(
                f"| {dataset_name} | {explicit:,} | {weak:,} | {not_res:,} | {total_mods:,} |"
            )

        # Data quality comparison
        report_lines.append("\n## Data Quality Comparison\n")
        report_lines.append("| Dataset | Missing Seq | Ambiguous Mod | Missing Label | Duplicates |")
        report_lines.append("|---------|-------------|---------------|---------------|------------|")

        for dataset_name, metrics in all_metrics.items():
            missing_seq = metrics.missing_sequence_count
            ambig_mod = metrics.ambiguous_modification_count
            missing_label = metrics.missing_label_count
            duplicates = metrics.duplicate_count

            report_lines.append(
                f"| {dataset_name} | {missing_seq:,} | {ambig_mod:,} | "
                f"{missing_label:,} | {duplicates:,} |"
            )

        # Overall recommendations
        report_lines.append("\n## Overall Assessment\n")

        viable_datasets = [
            name for name, metrics in all_metrics.items()
            if metrics.usable_total >= 100
        ]

        marginal_datasets = [
            name for name, metrics in all_metrics.items()
            if 50 <= metrics.usable_total < 100
        ]

        insufficient_datasets = [
            name for name, metrics in all_metrics.items()
            if metrics.usable_total < 50
        ]

        report_lines.append(f"**Viable for modeling**: {len(viable_datasets)} datasets")
        if viable_datasets:
            report_lines.append(f"  - {', '.join(viable_datasets)}")

        report_lines.append(f"\n**Marginal**: {len(marginal_datasets)} datasets")
        if marginal_datasets:
            report_lines.append(f"  - {', '.join(marginal_datasets)}")

        report_lines.append(f"\n**Insufficient**: {len(insufficient_datasets)} datasets")
        if insufficient_datasets:
            report_lines.append(f"  - {', '.join(insufficient_datasets)}")

        # Next steps
        report_lines.append("\n## Next Steps\n")

        if viable_datasets:
            report_lines.append(f"1. **Proceed to Stage 2 (Processing)** for: {', '.join(viable_datasets)}")
            report_lines.append("   - Implement dataset-specific parsers")
            report_lines.append("   - Refine anchor inference methods")
            report_lines.append("   - Generate processed datasets")

        if marginal_datasets:
            report_lines.append(f"\n2. **Review filtering criteria** for: {', '.join(marginal_datasets)}")
            report_lines.append("   - Assess if any exclusions can be relaxed")
            report_lines.append("   - Consider data augmentation strategies")

        if insufficient_datasets:
            report_lines.append(f"\n3. **Re-evaluate or exclude**: {', '.join(insufficient_datasets)}")
            report_lines.append("   - Seek additional data sources")
            report_lines.append("   - Consider alternative datasets")

        # Save report
        report_path = self.output_dir / "combined_summary.md"
        report_text = "\n".join(report_lines)

        with open(report_path, 'w') as f:
            f.write(report_text)

        return report_path
