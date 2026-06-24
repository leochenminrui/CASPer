#!/usr/bin/env python3
"""
Generate split summary reports.

Reads split metadata and generates markdown reports with:
- Split statistics
- Label distributions
- Edit family distributions
- Leakage risk analysis

Usage:
    python scripts/split_summary_report.py
    python scripts/split_summary_report.py --dataset CycPeptMPDB_PAMPA
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pem_schema import PEMSample
from src.data.serialization import load_samples_jsonl


def load_split_metadata(dataset_dir: Path) -> Dict[str, Any]:
    """Load split metadata for a dataset."""
    metadata_file = dataset_dir / 'split_metadata.json'
    if not metadata_file.exists():
        return {}

    with open(metadata_file, 'r') as f:
        return json.load(f)


def load_strategy_metadata(strategy_dir: Path) -> Dict[str, Any]:
    """Load metadata for a specific strategy."""
    metadata_file = strategy_dir / 'metadata.json'
    if not metadata_file.exists():
        return {}

    with open(metadata_file, 'r') as f:
        return json.load(f)


def generate_split_table(metadata: Dict[str, Any]) -> str:
    """Generate markdown table of split statistics."""
    lines = [
        "| Strategy | Feasible | Train | Val | Test | Total | Leakage Risk |",
        "|----------|----------|-------|-----|------|-------|--------------|",
    ]

    for strategy_name, info in metadata.get('strategies', {}).items():
        if info.get('feasible'):
            train = info.get('train_count', 0)
            val = info.get('val_count', 0)
            test = info.get('test_count', 0)
            total = train + val + test
            risk = info.get('leakage_risk', 'unknown')
            lines.append(
                f"| {strategy_name} | ✓ | {train} | {val} | {test} | {total} | {risk} |"
            )
        else:
            reason = info.get('reason', 'Unknown')
            lines.append(
                f"| {strategy_name} | ✗ | - | - | - | - | {reason} |"
            )

    return '\n'.join(lines)


def generate_leakage_analysis_section(
    dataset_dir: Path,
    strategy_name: str
) -> str:
    """Generate detailed leakage analysis section."""
    strategy_dir = dataset_dir / strategy_name
    metadata = load_strategy_metadata(strategy_dir)

    if not metadata or not metadata.get('feasible'):
        return f"*Strategy {strategy_name} not feasible*\n"

    leakage = metadata.get('leakage_analysis', {})

    lines = [
        f"### {strategy_name}\n",
        "#### Sequence Similarity",
        "",
        f"- **Max identity (train→test)**: {leakage.get('sequence_similarity', {}).get('max_identity', 0):.3f}",
        f"- **% test with >70% identity to train**: {leakage.get('sequence_similarity', {}).get('pct_high_similarity', 0):.1f}%",
        f"- **Mean test-train similarity**: {leakage.get('sequence_similarity', {}).get('mean_test_train', 0):.3f}",
        "",
        "#### Scaffold Overlap",
        "",
        f"- **Shared scaffolds**: {leakage.get('scaffold_overlap', {}).get('n_shared', 0)}",
        f"- **% test with shared scaffold**: {leakage.get('scaffold_overlap', {}).get('pct_test_shared', 0):.1f}%",
        "",
        "#### Edit Pattern Similarity",
        "",
        f"- **Cosine similarity**: {leakage.get('edit_pattern_similarity', {}).get('cosine_similarity', 0):.3f}",
        f"- **JS divergence**: {leakage.get('edit_pattern_similarity', {}).get('js_divergence', 0):.3f}",
        "",
        "#### Label Distribution",
        "",
    ]

    label_dist = leakage.get('label_distribution', {})
    lines.extend([
        f"- **KS statistic**: {label_dist.get('ks_statistic', 0):.4f} (p={label_dist.get('ks_pvalue', 1):.4f})",
        "",
        "| Split | Mean | Std |",
        "|-------|------|-----|",
        f"| Train | {label_dist.get('train', {}).get('mean', 0):.3f} | {label_dist.get('train', {}).get('std', 0):.3f} |",
        f"| Val   | {label_dist.get('val', {}).get('mean', 0):.3f} | {label_dist.get('val', {}).get('std', 0):.3f} |",
        f"| Test  | {label_dist.get('test', {}).get('mean', 0):.3f} | {label_dist.get('test', {}).get('std', 0):.3f} |",
        "",
        f"**Overall Risk**: `{leakage.get('overall_risk', 'unknown').upper()}`",
        "",
    ])

    return '\n'.join(lines)


def compute_edit_family_distribution(samples: List[PEMSample]) -> Dict[str, int]:
    """Compute edit family distribution."""
    family_counts = Counter()
    for sample in samples:
        for edit in sample.edits:
            family_counts[edit.edit_family] += 1
    return dict(family_counts)


def generate_edit_distribution_table(
    dataset_dir: Path,
    strategy_name: str
) -> str:
    """Generate edit family distribution table."""
    strategy_dir = dataset_dir / strategy_name

    # Load samples
    try:
        train_samples = load_samples_jsonl(strategy_dir / 'train.jsonl')
        val_samples = load_samples_jsonl(strategy_dir / 'val.jsonl')
        test_samples = load_samples_jsonl(strategy_dir / 'test.jsonl')
    except:
        return "*Data not available*\n"

    # Compute distributions
    train_dist = compute_edit_family_distribution(train_samples)
    val_dist = compute_edit_family_distribution(val_samples)
    test_dist = compute_edit_family_distribution(test_samples)

    # Get all families
    all_families = sorted(set(
        list(train_dist.keys()) +
        list(val_dist.keys()) +
        list(test_dist.keys())
    ))

    lines = [
        "| Edit Family | Train | Val | Test |",
        "|-------------|-------|-----|------|",
    ]

    for family in all_families:
        lines.append(
            f"| {family} | {train_dist.get(family, 0)} | "
            f"{val_dist.get(family, 0)} | {test_dist.get(family, 0)} |"
        )

    return '\n'.join(lines)


def generate_report_for_dataset(
    dataset_name: str,
    splits_dir: Path = Path('data/splits'),
    output_dir: Path = Path('reports/splits'),
) -> None:
    """Generate split summary report for a dataset."""
    dataset_dir = splits_dir / dataset_name

    if not dataset_dir.exists():
        print(f"No splits found for {dataset_name}")
        return

    # Load metadata
    metadata = load_split_metadata(dataset_dir)

    if not metadata:
        print(f"No metadata found for {dataset_name}")
        return

    # Start report
    lines = [
        f"# Split Summary: {dataset_name}",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Total Samples**: {metadata.get('total_samples', 0)}",
        "",
        f"**Recommended Strategy**: `{metadata.get('recommended_strategy', 'None')}`",
        "",
        "---",
        "",
        "## Split Statistics",
        "",
    ]

    # Add split table
    lines.append(generate_split_table(metadata))
    lines.append("")

    # Leakage analysis for each successful strategy
    lines.append("## Leakage Analysis")
    lines.append("")

    for strategy_name in metadata.get('successful_strategies', []):
        lines.append(generate_leakage_analysis_section(dataset_dir, strategy_name))

    # Edit distribution for recommended strategy
    recommended = metadata.get('recommended_strategy')
    if recommended:
        lines.append("## Edit Family Distribution")
        lines.append("")
        lines.append(f"Distribution for recommended strategy: `{recommended}`")
        lines.append("")
        lines.append(generate_edit_distribution_table(dataset_dir, recommended))
        lines.append("")

    # Strategy-specific details
    lines.append("## Strategy Details")
    lines.append("")

    for strategy_name in metadata.get('successful_strategies', []):
        strategy_dir = dataset_dir / strategy_name
        strategy_metadata = load_strategy_metadata(strategy_dir)

        if not strategy_metadata:
            continue

        lines.append(f"### {strategy_name}")
        lines.append("")

        # Parameters
        params = strategy_metadata.get('strategy_params', {})
        if params:
            lines.append("**Parameters**:")
            lines.append("")
            for key, value in params.items():
                lines.append(f"- `{key}`: {value}")
            lines.append("")

        # Strategy-specific stats
        if 'n_scaffolds' in strategy_metadata:
            lines.append(f"**Unique Scaffolds**: {strategy_metadata['n_scaffolds']}")
        if 'n_clusters' in strategy_metadata:
            lines.append(f"**Sequence Clusters**: {strategy_metadata['n_clusters']}")
        if 'n_edit_profiles' in strategy_metadata:
            lines.append(f"**Edit Profiles**: {strategy_metadata['n_edit_profiles']}")
        if 'n_derivative_groups' in strategy_metadata:
            lines.append(f"**Derivative Groups**: {strategy_metadata['n_derivative_groups']}")

        lines.append("")

    # Write report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f'{dataset_name}_split_summary.md'

    with open(report_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Generated report: {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate split summary reports'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        help='Dataset name or "all" (default: all)'
    )
    parser.add_argument(
        '--splits-dir',
        type=Path,
        default=Path('data/splits'),
        help='Splits directory (default: data/splits)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports/splits'),
        help='Output directory (default: reports/splits)'
    )

    args = parser.parse_args()

    # Determine datasets
    if args.dataset == 'all':
        # Find all datasets in splits dir
        if args.splits_dir.exists():
            datasets = [
                d.name for d in args.splits_dir.iterdir()
                if d.is_dir() and (d / 'split_metadata.json').exists()
            ]
        else:
            print(f"Splits directory not found: {args.splits_dir}")
            return
    else:
        datasets = [args.dataset]

    # Generate reports
    for dataset_name in datasets:
        print(f"\nGenerating report for {dataset_name}...")
        generate_report_for_dataset(
            dataset_name,
            splits_dir=args.splits_dir,
            output_dir=args.output_dir,
        )

    print(f"\n✓ Reports saved to {args.output_dir}")


if __name__ == '__main__':
    main()
