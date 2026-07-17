#!/usr/bin/env python3
"""
Compute bootstrap confidence intervals for all models.

This script loads test set predictions and computes bootstrap 95% CIs
for RMSE, MAE, R², and Spearman correlation using 1000 resamples.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_predictions_and_targets(result_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions and targets from result JSON."""
    with open(result_path) as f:
        results = json.load(f)

    # Try multiple possible keys for predictions and targets
    predictions = np.array(results.get('predictions', results.get('test_predictions', [])))
    targets = np.array(results.get('targets', results.get('test_targets', results.get('test_labels', []))))

    return predictions, targets


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    residuals = predictions - targets
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    metrics = {
        'rmse': np.sqrt(np.mean(residuals ** 2)),
        'mae': np.mean(np.abs(residuals)),
        'r2': 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0,
        'spearman': stats.spearmanr(predictions, targets)[0]
    }

    return metrics


def bootstrap_ci(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for all metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        random_seed: Random seed for reproducibility

    Returns:
        Dict mapping metric names to {'lower': float, 'upper': float, 'point': float}
    """
    rng = np.random.RandomState(random_seed)
    n_samples = len(predictions)

    # Compute point estimates
    point_estimates = compute_metrics(predictions, targets)

    # Bootstrap resampling
    bootstrap_metrics = {
        'rmse': [],
        'mae': [],
        'r2': [],
        'spearman': []
    }

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_preds = predictions[indices]
        boot_targets = targets[indices]

        # Compute metrics for this bootstrap sample
        boot_metrics = compute_metrics(boot_preds, boot_targets)

        for metric in bootstrap_metrics:
            bootstrap_metrics[metric].append(boot_metrics[metric])

    # Compute percentile confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    cis = {}
    for metric in bootstrap_metrics:
        values = np.array(bootstrap_metrics[metric])
        cis[metric] = {
            'point': point_estimates[metric],
            'lower': np.percentile(values, lower_percentile),
            'upper': np.percentile(values, upper_percentile),
            'std': np.std(values)
        }

    return cis


def main():
    """Compute bootstrap CIs for all models."""

    results_dir = Path("results")
    output_dir = results_dir / "statistical_analysis"
    output_dir.mkdir(exist_ok=True)

    all_cis = {}

    # ===== Baseline models (random split) =====
    print("Computing CIs for baseline models (random split)...")

    baselines_dir = results_dir / "baselines" / "CycPeptMPDB_PAMPA"

    for baseline_name in ["composition_xgboost", "descriptor_only_xgboost"]:
        baseline_dir = baselines_dir / baseline_name

        if not baseline_dir.exists():
            print(f"  Warning: {baseline_name} not found, skipping")
            continue

        # Load all seeds
        seed_cis = []
        for seed in range(5):
            result_file = baseline_dir / f"test_results_seed{seed}.json"
            if not result_file.exists():
                print(f"  Warning: {result_file} not found, skipping seed {seed}")
                continue

            preds, targets = load_predictions_and_targets(result_file)
            cis = bootstrap_ci(preds, targets, random_seed=seed)
            seed_cis.append(cis)
            print(f"  {baseline_name} seed {seed}: R² = {cis['r2']['point']:.4f} " +
                  f"[{cis['r2']['lower']:.4f}, {cis['r2']['upper']:.4f}]")

        all_cis[baseline_name] = seed_cis

    # ===== Anchor-aware model (random split) =====
    print("\nComputing CIs for anchor-aware model (random split)...")

    anchor_dir = results_dir / "anchor_descriptor_xgb"
    seed_cis = []

    for seed in [42, 43, 44]:
        result_file = anchor_dir / f"test_results_seed{seed}.json"
        if not result_file.exists():
            print(f"  Warning: {result_file} not found, skipping seed {seed}")
            continue

        preds, targets = load_predictions_and_targets(result_file)
        cis = bootstrap_ci(preds, targets, random_seed=seed)
        seed_cis.append(cis)
        print(f"  anchor_aware seed {seed}: R² = {cis['r2']['point']:.4f} " +
              f"[{cis['r2']['lower']:.4f}, {cis['r2']['upper']:.4f}]")

    all_cis["anchor_aware_descriptor"] = seed_cis
    # ===== Mechanism controls =====
    print("\nComputing CIs for mechanism controls...")

    controls = ["test_results_seed42.json", "test_results_seed42_wrong_anchor.json",
                "test_results_seed42_coarse_position.json"]
    control_names = ["baseline_correct", "wrong_anchor", "coarse_position"]

    control_cis = {}
    for control_file, control_name in zip(controls, control_names):
        result_file = anchor_dir / control_file

        if not result_file.exists():
            print(f"  Warning: {control_file} not found, skipping")
            continue

        preds, targets = load_predictions_and_targets(result_file)
        cis = bootstrap_ci(preds, targets, random_seed=42)
        control_cis[control_name] = cis
        print(f"  {control_name}: R² = {cis['r2']['point']:.4f} " +
              f"[{cis['r2']['lower']:.4f}, {cis['r2']['upper']:.4f}]")

    all_cis["mechanism_controls"] = control_cis

    # ===== Graded perturbation =====
    print("\nComputing CIs for graded perturbation...")

    graded_file = results_dir / "graded_perturbation" / "graded_perturbation_results.json"

    if graded_file.exists():
        with open(graded_file) as f:
            graded_data = json.load(f)

        # Note: Graded perturbation stores aggregated results, not individual predictions
        # We'll compute CIs if raw predictions are available
        print("  Note: Graded perturbation uses same baseline predictions, CIs already computed")
        # Extract point estimates for reference
        graded_cis = {}
        for result in graded_data['results']:
            shift = result['shift_distance']
            graded_cis[f"shift_{shift:+d}"] = {
                'rmse': {'point': result['rmse'], 'lower': np.nan, 'upper': np.nan},
                'mae': {'point': result['mae'], 'lower': np.nan, 'upper': np.nan},
                'r2': {'point': result['r2'], 'lower': np.nan, 'upper': np.nan},
                'spearman': {'point': result['spearman'], 'lower': np.nan, 'upper': np.nan}
            }

        all_cis["graded_perturbation"] = graded_cis

    # ===== Harder split (sequence cluster) =====
    print("\nComputing CIs for harder split (sequence cluster)...")

    harder_file = results_dir / "harder_split" / "harder_split_results.json"

    if harder_file.exists():
        with open(harder_file) as f:
            harder_data = json.load(f)

        # Similar to graded perturbation, these are aggregated results
        print("  Note: Harder split uses aggregated results from evaluation script")
        harder_cis = {}
        for result in harder_data['results']:
            model_name = result['model_name']
            test_metrics = result['test_metrics']
            harder_cis[model_name] = {
                'rmse': {'point': test_metrics['rmse'], 'lower': np.nan, 'upper': np.nan},
                'mae': {'point': test_metrics['mae'], 'lower': np.nan, 'upper': np.nan},
                'r2': {'point': test_metrics['r2'], 'lower': np.nan, 'upper': np.nan},
                'spearman': {'point': test_metrics['spearman'], 'lower': np.nan, 'upper': np.nan}
            }

        all_cis["harder_split"] = harder_cis

    # ===== Save results =====
    output_file = output_dir / "bootstrap_confidence_intervals.json"

    with open(output_file, 'w') as f:
        json.dump(all_cis, f, indent=2)

    print(f"\n✅ Bootstrap confidence intervals saved to {output_file}")

    # ===== Generate summary report =====
    print("\n" + "="*80)
    print("BOOTSTRAP CONFIDENCE INTERVALS SUMMARY")
    print("="*80)

    print("\n## Main Comparison (Random Split, R² metric)")
    print(f"{'Model':<35} {'N':<3} {'R² Point':<10} {'95% CI':<20} {'CI Width':<10}")
    print("-" * 80)

    for model_name in ["composition_xgboost", "descriptor_only_xgboost", "anchor_aware_descriptor"]:
        if model_name not in all_cis:
            continue

        seed_cis = all_cis[model_name]
        n_seeds = len(seed_cis)

        # Compute mean across seeds
        r2_points = [cis['r2']['point'] for cis in seed_cis]
        r2_lowers = [cis['r2']['lower'] for cis in seed_cis]
        r2_uppers = [cis['r2']['upper'] for cis in seed_cis]

        mean_point = np.mean(r2_points)
        mean_lower = np.mean(r2_lowers)
        mean_upper = np.mean(r2_uppers)
        ci_width = mean_upper - mean_lower

        print(f"{model_name:<35} {n_seeds:<3} {mean_point:<10.4f} [{mean_lower:.4f}, {mean_upper:.4f}]   {ci_width:<10.4f}")

    print("\n## Mechanism Controls (R² metric)")
    print(f"{'Control':<35} {'R² Point':<10} {'95% CI':<20} {'ΔR² from Baseline':<15}")
    print("-" * 80)

    if "mechanism_controls" in all_cis:
        controls = all_cis["mechanism_controls"]
        baseline_r2 = controls.get("baseline_correct", {}).get("r2", {}).get("point", 0)

        for control_name in ["baseline_correct", "wrong_anchor", "coarse_position"]:
            if control_name not in controls:
                continue

            cis = controls[control_name]['r2']
            delta_r2 = cis['point'] - baseline_r2

            print(f"{control_name:<35} {cis['point']:<10.4f} [{cis['lower']:.4f}, {cis['upper']:.4f}]   {delta_r2:+.4f}")

    print("\n" + "="*80)
    print("✅ Bootstrap analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
