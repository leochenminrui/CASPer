#!/usr/bin/env python3
"""
Evaluate baseline models and generate comparison reports.

Supports statistical significance testing and benchmark table generation.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baselines.base import BaselineResult


def load_baseline_results(results_dir: Path) -> Dict[str, BaselineResult]:
    """
    Load all baseline results from directory.

    Args:
        results_dir: Directory containing baseline results

    Returns:
        Dictionary mapping baseline name to aggregated result
    """
    results = {}

    for baseline_dir in results_dir.iterdir():
        if not baseline_dir.is_dir():
            continue

        # Load aggregated results
        agg_path = baseline_dir / "test_results_aggregated.json"
        if agg_path.exists():
            result = BaselineResult.load(agg_path)
            results[baseline_dir.name] = result

    return results


def compute_significance(
    results1: List[float],
    results2: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute statistical significance between two sets of results.

    Args:
        results1: Results from first model (per seed)
        results2: Results from second model (per seed)
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(results1, results2)

    significant = p_value < alpha

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': significant,
        'better': 'model1' if np.mean(results1) < np.mean(results2) else 'model2',
    }


def generate_comparison_table(
    results: Dict[str, BaselineResult],
    metric: str = 'rmse',
) -> pd.DataFrame:
    """
    Generate comparison table for baselines.

    Args:
        results: Dictionary of baseline results
        metric: Metric to compare

    Returns:
        DataFrame with comparison
    """
    rows = []

    for name, result in results.items():
        # Get category from result
        category = "Unknown"
        if "composition" in name or "lm" in name:
            category = "Sequence-Only"
        elif "descriptor" in name or "anchor" in name or "late" in name or "fusion" in name:
            category = "Chemistry-Aware"
        elif "delta" in name or "ranking" in name:
            category = "Paired"

        # Get metrics
        mean_val = result.metrics_mean[metric]
        std_val = result.metrics_std[metric]
        ci_lower = result.metrics_ci_lower[metric]
        ci_upper = result.metrics_ci_upper[metric]

        rows.append({
            'Model': name,
            'Category': category,
            f'{metric.upper()} Mean': mean_val,
            f'{metric.upper()} Std': std_val,
            'CI Lower': ci_lower,
            'CI Upper': ci_upper,
            f'{metric.upper()} ± Std': f"{mean_val:.4f} ± {std_val:.4f}",
        })

    df = pd.DataFrame(rows)

    # Sort by mean value
    df = df.sort_values(f'{metric.upper()} Mean')

    return df


def generate_markdown_table(
    results: Dict[str, BaselineResult],
    output_path: Path,
):
    """
    Generate markdown comparison table.

    Args:
        results: Dictionary of baseline results
        output_path: Output file path
    """
    lines = []

    # Header
    lines.append("# Baseline Comparison\n")
    lines.append(f"**Total baselines**: {len(results)}\n")
    lines.append("")

    # Main comparison table
    lines.append("## Results Summary\n")
    lines.append("| Model | Category | RMSE ↓ | MAE ↓ | R² ↑ | Spearman ↑ |")
    lines.append("|-------|----------|---------|-------|------|------------|")

    # Group by category
    categories = {
        'Sequence-Only': [],
        'Chemistry-Aware': [],
        'Paired': [],
    }

    for name, result in results.items():
        if "composition" in name or "lm" in name:
            category = "Sequence-Only"
        elif "descriptor" in name or "anchor" in name or "late" in name or "fusion" in name:
            category = "Chemistry-Aware"
        elif "delta" in name or "ranking" in name:
            category = "Paired"
        else:
            category = "Other"

        if category in categories:
            categories[category].append((name, result))
        else:
            categories[category] = [(name, result)]

    # Add rows by category
    for category, items in categories.items():
        if not items:
            continue

        lines.append(f"| **{category}** | | | | | |")

        for name, result in items:
            metrics = result.metrics_mean
            stds = result.metrics_std

            rmse_str = f"{metrics.get('rmse', 0):.4f} ± {stds.get('rmse', 0):.4f}"
            mae_str = f"{metrics.get('mae', 0):.4f} ± {stds.get('mae', 0):.4f}"
            r2_str = f"{metrics.get('r2', 0):.4f} ± {stds.get('r2', 0):.4f}"
            spearman_str = f"{metrics.get('spearman', 0):.4f} ± {stds.get('spearman', 0):.4f}"

            lines.append(
                f"| {name} | {category[:4]} | {rmse_str} | {mae_str} | {r2_str} | {spearman_str} |"
            )

    lines.append("")

    # Detailed metrics
    lines.append("## Detailed Metrics\n")

    for metric in ['rmse', 'mae', 'r2', 'spearman']:
        lines.append(f"\n### {metric.upper()}\n")
        lines.append("| Model | Mean | Std | 95% CI |")
        lines.append("|-------|------|-----|--------|")

        # Sort by mean
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].metrics_mean.get(metric, 0),
            reverse=(metric in ['r2', 'spearman'])  # Higher is better
        )

        for name, result in sorted_results:
            mean_val = result.metrics_mean.get(metric, 0)
            std_val = result.metrics_std.get(metric, 0)
            ci_lower = result.metrics_ci_lower.get(metric, 0)
            ci_upper = result.metrics_ci_upper.get(metric, 0)

            lines.append(
                f"| {name} | {mean_val:.4f} | {std_val:.4f} | [{ci_lower:.4f}, {ci_upper:.4f}] |"
            )

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Markdown table saved to: {output_path}")


def generate_benchmark_csv(
    results: Dict[str, BaselineResult],
    output_path: Path,
):
    """
    Generate CSV benchmark table.

    Args:
        results: Dictionary of baseline results
        output_path: Output file path
    """
    df = generate_comparison_table(results, metric='rmse')

    # Add all metrics
    for name, result in results.items():
        idx = df[df['Model'] == name].index
        if len(idx) > 0:
            idx = idx[0]
            for metric in ['mae', 'r2', 'spearman']:
                df.loc[idx, f'{metric.upper()}'] = result.metrics_mean.get(metric, 0)
                df.loc[idx, f'{metric.upper()} Std'] = result.metrics_std.get(metric, 0)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"CSV table saved to: {output_path}")


def evaluate_baseline(
    model_path: Path,
    split: str = "test",
):
    """
    Evaluate a single baseline.

    Args:
        model_path: Path to baseline directory
        split: Which split to evaluate
    """
    # Load aggregated results
    result_path = model_path / f"{split}_results_aggregated.json"

    if not result_path.exists():
        print(f"ERROR: Results not found at {result_path}")
        return

    result = BaselineResult.load(result_path)

    # Print results
    print(f"\nResults for {model_path.name} on {split} split")
    print("="*80)

    metrics = result.metrics_mean
    stds = result.metrics_std
    cis_lower = result.metrics_ci_lower
    cis_upper = result.metrics_ci_upper

    for metric in ['rmse', 'mae', 'r2', 'spearman']:
        if metric in metrics:
            print(f"{metric.upper():>10}: {metrics[metric]:.4f} ± {stds[metric]:.4f} "
                  f"[{cis_lower[metric]:.4f}, {cis_upper[metric]:.4f}]")


def compare_all_baselines(
    dataset: str,
    split: str,
    results_dir: Path,
    output_dir: Path,
):
    """
    Compare all baselines for a dataset.

    Args:
        dataset: Dataset name
        split: Split name
        results_dir: Directory with baseline results
        output_dir: Output directory for reports
    """
    # Load all results
    dataset_results_dir = results_dir / dataset
    results = load_baseline_results(dataset_results_dir)

    if not results:
        print(f"No results found in {dataset_results_dir}")
        return

    print(f"\nLoaded {len(results)} baseline results for {dataset}")

    # Generate reports
    report_dir = output_dir / dataset

    # Markdown table
    generate_markdown_table(
        results,
        report_dir / f"{split}_baseline_comparison.md",
    )

    # CSV table
    generate_benchmark_csv(
        results,
        report_dir / f"{split}_benchmark_table.csv",
    )

    # Detailed JSON
    detailed_results = {}
    for name, result in results.items():
        detailed_results[name] = result.to_dict()

    json_path = report_dir / f"{split}_detailed_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"Detailed JSON saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models and generate reports"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to baseline model directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (for comparison mode)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to evaluate"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/baselines"),
        help="Directory with baseline results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/baselines"),
        help="Output directory for reports"
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all baselines for dataset"
    )

    args = parser.parse_args()

    if args.compare_all:
        if not args.dataset:
            print("ERROR: Must specify --dataset for comparison")
            return

        compare_all_baselines(
            args.dataset,
            args.split,
            args.results_dir,
            args.output_dir,
        )
    else:
        if not args.model:
            print("ERROR: Must specify --model or use --compare-all")
            return

        evaluate_baseline(args.model, args.split)


if __name__ == "__main__":
    main()
