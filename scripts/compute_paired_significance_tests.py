#!/usr/bin/env python3
"""
Compute paired significance tests for model comparisons.

This script performs:
1. Paired Wilcoxon signed-rank tests (non-parametric, robust)
2. Paired t-tests (parametric, assume normality)
3. Effect size quantification (Cohen's d)
4. Monotonic trend tests for graded perturbation
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

    # Try multiple possible keys
    predictions = np.array(results.get('predictions', results.get('test_predictions', [])))
    targets = np.array(results.get('targets', results.get('test_targets', results.get('test_labels', []))))

    return predictions, targets


def compute_squared_errors(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute per-sample squared errors."""
    return (predictions - targets) ** 2


def compute_absolute_errors(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute per-sample absolute errors."""
    return np.abs(predictions - targets)


def paired_wilcoxon_test(errors1: np.ndarray, errors2: np.ndarray) -> Dict[str, float]:
    """
    Perform paired Wilcoxon signed-rank test.

    Args:
        errors1: Per-sample errors for model 1
        errors2: Per-sample errors for model 2

    Returns:
        Dict with statistic, p-value, and interpretation
    """
    statistic, pvalue = stats.wilcoxon(errors1, errors2, alternative='two-sided')

    return {
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant_at_0.05': bool(pvalue < 0.05),
        'significant_at_0.01': bool(pvalue < 0.01),
        'significant_at_0.001': bool(pvalue < 0.001)
    }


def paired_ttest(errors1: np.ndarray, errors2: np.ndarray) -> Dict[str, float]:
    """Perform paired t-test."""
    statistic, pvalue = stats.ttest_rel(errors1, errors2, alternative='two-sided')

    return {
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant_at_0.05': bool(pvalue < 0.05),
        'significant_at_0.01': bool(pvalue < 0.01),
        'significant_at_0.001': bool(pvalue < 0.001)
    }


def cohens_d(errors1: np.ndarray, errors2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    Cohen's d interpretation:
    - Small effect: d ~ 0.2
    - Medium effect: d ~ 0.5
    - Large effect: d ~ 0.8
    """
    diff = errors1 - errors2
    d = np.mean(diff) / np.std(diff, ddof=1)

    return float(d)


def main():
    """Compute paired significance tests for all key comparisons."""

    results_dir = Path("results")
    output_dir = results_dir / "statistical_analysis"
    output_dir.mkdir(exist_ok=True)

    all_tests = {}

    print("="*80)
    print("PAIRED SIGNIFICANCE TESTS")
    print("="*80)

    # ===== Comparison 1: Anchor-aware vs Descriptor-only =====
    print("\n## Comparison 1: Anchor-aware vs Descriptor-only (Random Split)")
    print("-" * 80)

    # Load anchor-aware (seed 42 as representative)
    anchor_file = results_dir / "anchor_descriptor_xgb" / "test_results_seed42.json"
    anchor_preds, anchor_targets = load_predictions_and_targets(anchor_file)

    # Load descriptor-only (seed 0 as representative)
    descriptor_file = results_dir / "baselines" / "CycPeptMPDB_PAMPA" / "descriptor_only_xgboost" / "test_results_seed0.json"
    descriptor_preds, descriptor_targets = load_predictions_and_targets(descriptor_file)

    # Verify same targets
    assert np.allclose(anchor_targets, descriptor_targets), "Targets must match for paired test"

    # Compute errors
    anchor_sq_errors = compute_squared_errors(anchor_preds, anchor_targets)
    descriptor_sq_errors = compute_squared_errors(descriptor_preds, descriptor_targets)

    anchor_abs_errors = compute_absolute_errors(anchor_preds, anchor_targets)
    descriptor_abs_errors = compute_absolute_errors(descriptor_preds, descriptor_targets)

    # Wilcoxon test on squared errors (tests if R² differs)
    wilcoxon_sq = paired_wilcoxon_test(descriptor_sq_errors, anchor_sq_errors)
    ttest_sq = paired_ttest(descriptor_sq_errors, anchor_sq_errors)
    cohens_sq = cohens_d(descriptor_sq_errors, anchor_sq_errors)

    # Wilcoxon test on absolute errors (tests if MAE differs)
    wilcoxon_abs = paired_wilcoxon_test(descriptor_abs_errors, anchor_abs_errors)
    ttest_abs = paired_ttest(descriptor_abs_errors, anchor_abs_errors)
    cohens_abs = cohens_d(descriptor_abs_errors, anchor_abs_errors)

    all_tests['anchor_vs_descriptor'] = {
        'description': 'Anchor-aware vs Descriptor-only (lower error is better)',
        'n_samples': len(anchor_targets),
        'squared_error': {
            'wilcoxon': wilcoxon_sq,
            'ttest': ttest_sq,
            'cohens_d': cohens_sq
        },
        'absolute_error': {
            'wilcoxon': wilcoxon_abs,
            'ttest': ttest_abs,
            'cohens_d': cohens_abs
        }
    }

    print(f"  N samples: {len(anchor_targets)}")
    print(f"  Squared Error (MSE):")
    print(f"    Wilcoxon p-value: {wilcoxon_sq['pvalue']:.6f} {'***' if wilcoxon_sq['significant_at_0.001'] else '**' if wilcoxon_sq['significant_at_0.01'] else '*' if wilcoxon_sq['significant_at_0.05'] else 'ns'}")
    print(f"    Paired t-test p-value: {ttest_sq['pvalue']:.6f}")
    print(f"    Cohen's d: {cohens_sq:.4f} ({'large' if abs(cohens_sq) > 0.8 else 'medium' if abs(cohens_sq) > 0.5 else 'small' if abs(cohens_sq) > 0.2 else 'negligible'} effect)")
    print(f"  Absolute Error (MAE):")
    print(f"    Wilcoxon p-value: {wilcoxon_abs['pvalue']:.6f} {'***' if wilcoxon_abs['significant_at_0.001'] else '**' if wilcoxon_abs['significant_at_0.01'] else '*' if wilcoxon_abs['significant_at_0.05'] else 'ns'}")
    print(f"    Cohen's d: {cohens_abs:.4f}")

    # ===== Comparison 2: Correct anchor vs Wrong anchor =====
    print("\n## Comparison 2: Correct Anchor vs Wrong Anchor (Mechanism Control)")
    print("-" * 80)

    correct_file = results_dir / "anchor_descriptor_xgb" / "test_results_seed42.json"
    wrong_file = results_dir / "anchor_descriptor_xgb" / "test_results_seed42_wrong_anchor.json"

    correct_preds, correct_targets = load_predictions_and_targets(correct_file)
    wrong_preds, wrong_targets = load_predictions_and_targets(wrong_file)

    assert np.allclose(correct_targets, wrong_targets), "Targets must match"

    correct_sq_errors = compute_squared_errors(correct_preds, correct_targets)
    wrong_sq_errors = compute_squared_errors(wrong_preds, wrong_targets)

    wilcoxon_control = paired_wilcoxon_test(correct_sq_errors, wrong_sq_errors)
    ttest_control = paired_ttest(correct_sq_errors, wrong_sq_errors)
    cohens_control = cohens_d(correct_sq_errors, wrong_sq_errors)

    all_tests['correct_vs_wrong_anchor'] = {
        'description': 'Correct anchor vs Wrong anchor (lower error is better)',
        'n_samples': len(correct_targets),
        'squared_error': {
            'wilcoxon': wilcoxon_control,
            'ttest': ttest_control,
            'cohens_d': cohens_control
        }
    }

    print(f"  N samples: {len(correct_targets)}")
    print(f"  Squared Error (MSE):")
    print(f"    Wilcoxon p-value: {wilcoxon_control['pvalue']:.6f} {'***' if wilcoxon_control['significant_at_0.001'] else '**' if wilcoxon_control['significant_at_0.01'] else '*' if wilcoxon_control['significant_at_0.05'] else 'ns'}")
    print(f"    Paired t-test p-value: {ttest_control['pvalue']:.6f}")
    print(f"    Cohen's d: {cohens_control:.4f} ({'large' if abs(cohens_control) > 0.8 else 'medium' if abs(cohens_control) > 0.5 else 'small' if abs(cohens_control) > 0.2 else 'negligible'} effect)")

    # ===== Comparison 3: Correct anchor vs Coarse position =====
    print("\n## Comparison 3: Correct Anchor vs Coarse Position (Mechanism Control)")
    print("-" * 80)

    coarse_file = results_dir / "anchor_descriptor_xgb" / "test_results_seed42_coarse_position.json"
    coarse_preds, coarse_targets = load_predictions_and_targets(coarse_file)

    assert np.allclose(correct_targets, coarse_targets), "Targets must match"

    coarse_sq_errors = compute_squared_errors(coarse_preds, coarse_targets)

    wilcoxon_coarse = paired_wilcoxon_test(correct_sq_errors, coarse_sq_errors)
    ttest_coarse = paired_ttest(correct_sq_errors, coarse_sq_errors)
    cohens_coarse = cohens_d(correct_sq_errors, coarse_sq_errors)

    all_tests['correct_vs_coarse_position'] = {
        'description': 'Correct anchor vs Coarse position (lower error is better)',
        'n_samples': len(correct_targets),
        'squared_error': {
            'wilcoxon': wilcoxon_coarse,
            'ttest': ttest_coarse,
            'cohens_d': cohens_coarse
        }
    }

    print(f"  N samples: {len(correct_targets)}")
    print(f"  Squared Error (MSE):")
    print(f"    Wilcoxon p-value: {wilcoxon_coarse['pvalue']:.6f} {'***' if wilcoxon_coarse['significant_at_0.001'] else '**' if wilcoxon_coarse['significant_at_0.01'] else '*' if wilcoxon_coarse['significant_at_0.05'] else 'ns'}")
    print(f"    Paired t-test p-value: {ttest_coarse['pvalue']:.6f}")
    print(f"    Cohen's d: {cohens_coarse:.4f} ({'large' if abs(cohens_coarse) > 0.8 else 'medium' if abs(cohens_coarse) > 0.5 else 'small' if abs(cohens_coarse) > 0.2 else 'negligible'} effect)")

    # ===== Comparison 4: Chemistry matters (Descriptor-only vs Composition) =====
    print("\n## Comparison 4: Descriptor-only vs Composition (Chemistry Contribution)")
    print("-" * 80)

    composition_file = results_dir / "baselines" / "CycPeptMPDB_PAMPA" / "composition_xgboost" / "test_results_seed0.json"
    composition_preds, composition_targets = load_predictions_and_targets(composition_file)

    assert np.allclose(descriptor_targets, composition_targets), "Targets must match"

    composition_sq_errors = compute_squared_errors(composition_preds, composition_targets)

    wilcoxon_chem = paired_wilcoxon_test(composition_sq_errors, descriptor_sq_errors)
    ttest_chem = paired_ttest(composition_sq_errors, descriptor_sq_errors)
    cohens_chem = cohens_d(composition_sq_errors, descriptor_sq_errors)

    all_tests['descriptor_vs_composition'] = {
        'description': 'Descriptor-only vs Composition (lower error is better, tests chemistry contribution)',
        'n_samples': len(composition_targets),
        'squared_error': {
            'wilcoxon': wilcoxon_chem,
            'ttest': ttest_chem,
            'cohens_d': cohens_chem
        }
    }

    print(f"  N samples: {len(composition_targets)}")
    print(f"  Squared Error (MSE):")
    print(f"    Wilcoxon p-value: {wilcoxon_chem['pvalue']:.6f} {'***' if wilcoxon_chem['significant_at_0.001'] else '**' if wilcoxon_chem['significant_at_0.01'] else '*' if wilcoxon_chem['significant_at_0.05'] else 'ns'}")
    print(f"    Paired t-test p-value: {ttest_chem['pvalue']:.6f}")
    print(f"    Cohen's d: {cohens_chem:.4f} ({'large' if abs(cohens_chem) > 0.8 else 'medium' if abs(cohens_chem) > 0.5 else 'small' if abs(cohens_chem) > 0.2 else 'negligible'} effect)")

    # ===== Graded Perturbation: Monotonic Trend Test =====
    print("\n## Graded Perturbation: Monotonic Degradation Trend Test")
    print("-" * 80)

    # Load graded perturbation results
    graded_file = results_dir / "graded_perturbation" / "graded_perturbation_results.json"

    with open(graded_file) as f:
        graded_data = json.load(f)

    # Extract R² values by absolute shift distance
    shifts = []
    r2_values = []

    for result in graded_data['results']:
        abs_shift = abs(result['shift_distance'])
        r2 = result['r2']

        shifts.append(abs_shift)
        r2_values.append(r2)

    # Sort by shift distance
    sorted_indices = np.argsort(shifts)
    shifts = np.array(shifts)[sorted_indices]
    r2_values = np.array(r2_values)[sorted_indices]

    # Spearman correlation: expect negative correlation (larger shift → worse R²)
    spearman_rho, spearman_p = stats.spearmanr(shifts, r2_values)

    # Jonckheere-Terpstra trend test (non-parametric test for ordered alternatives)
    # Not available in scipy, but we can use Spearman as proxy

    all_tests['graded_perturbation_trend'] = {
        'description': 'Monotonic degradation with increasing anchor shift distance',
        'n_points': int(len(shifts)),
        'spearman_correlation': {
            'rho': float(spearman_rho),
            'pvalue': float(spearman_p),
            'significant_at_0.05': bool(spearman_p < 0.05),
            'significant_at_0.01': bool(spearman_p < 0.01),
            'significant_at_0.001': bool(spearman_p < 0.001)
        },
        'interpretation': 'Negative correlation indicates degradation with increasing shift'
    }

    print(f"  N data points: {len(shifts)}")
    print(f"  Spearman correlation (shift distance vs R²):")
    print(f"    ρ = {spearman_rho:.4f}")
    print(f"    p-value: {spearman_p:.6f} {'***' if spearman_p < 0.001 else '**' if spearman_p < 0.01 else '*' if spearman_p < 0.05 else 'ns'}")
    print(f"    Interpretation: {'Significant monotonic degradation' if spearman_p < 0.05 else 'No significant trend'}")

    # ===== Save results =====
    output_file = output_dir / "paired_significance_tests.json"

    with open(output_file, 'w') as f:
        json.dump(all_tests, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Paired significance tests saved to {output_file}")
    print("="*80)

    # ===== Summary Table =====
    print("\n" + "="*80)
    print("SUMMARY OF SIGNIFICANCE TESTS")
    print("="*80)

    print(f"\n{'Comparison':<45} {'p-value':<12} {'Significance':<15} {'Effect Size'}")
    print("-" * 80)

    comparisons = [
        ('Anchor-aware vs Descriptor-only', all_tests['anchor_vs_descriptor']['squared_error']),
        ('Correct vs Wrong anchor', all_tests['correct_vs_wrong_anchor']['squared_error']),
        ('Correct vs Coarse position', all_tests['correct_vs_coarse_position']['squared_error']),
        ('Descriptor-only vs Composition', all_tests['descriptor_vs_composition']['squared_error']),
    ]

    for name, test_data in comparisons:
        pvalue = test_data['wilcoxon']['pvalue']
        cohens = test_data['cohens_d']

        sig_level = '***' if pvalue < 0.001 else '**' if pvalue < 0.01 else '*' if pvalue < 0.05 else 'ns'
        effect_size = 'large' if abs(cohens) > 0.8 else 'medium' if abs(cohens) > 0.5 else 'small' if abs(cohens) > 0.2 else 'negligible'

        print(f"{name:<45} {pvalue:<12.6f} {sig_level:<15} d={cohens:+.4f} ({effect_size})")

    print("\n" + "="*80)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    print("Effect sizes: |d| > 0.8 large, |d| > 0.5 medium, |d| > 0.2 small")
    print("="*80)


if __name__ == "__main__":
    main()
