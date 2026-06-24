#!/usr/bin/env python3
"""
Evaluate models on harder generalization split (sequence cluster split).

Tests whether models can generalize to novel sequence contexts.
"""

import argparse
import json
import csv
from pathlib import Path
import sys
import numpy as np
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import load_pem_dataset
from baselines.base import BaselineConfig
from baselines.chemistry_aware.anchor_aware_descriptor import AnchorAwareDescriptorBaseline
from baselines.sequence_only.composition_baseline import CompositionBaseline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'spearman': float(spearman),
    }


def train_and_evaluate_model(
    model_name: str,
    train_samples,
    val_samples,
    test_samples,
    seed: int
) -> Dict[str, Any]:
    """Train and evaluate a single model."""

    print(f"\nTraining: {model_name}")

    if model_name == "sequence_composition":
        config = BaselineConfig(
            name="composition_xgboost",
            category="sequence_only",
            version="1.0.0",
            model_type="xgboost",
            featurizer="composition",
            model_params={
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
            },
            featurizer_params={},
            random_seed=seed,
        )
        model = CompositionBaseline(config)

    elif model_name == "descriptor_only":
        # Use descriptor-only by setting ablation_mode
        config = BaselineConfig(
            name="descriptor_only",
            category="chemistry_aware",
            version="1.0.0",
            model_type="xgboost",
            featurizer="anchor_aware_descriptor",
            model_params={
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            },
            featurizer_params={
                'descriptor_set': 'basic',
                'ablation_mode': 'chemistry_only',
            },
            random_seed=seed,
        )
        model = AnchorAwareDescriptorBaseline(config)

    elif model_name == "anchor_aware_descriptor":
        config = BaselineConfig(
            name="anchor_aware_descriptor",
            category="chemistry_aware",
            version="1.0.0",
            model_type="xgboost",
            featurizer="anchor_aware_descriptor",
            model_params={
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            },
            featurizer_params={
                'descriptor_set': 'basic',
                'ablation_mode': 'full',
            },
            random_seed=seed,
        )
        model = AnchorAwareDescriptorBaseline(config)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train
    train_history = model.train(train_samples, val_samples)
    print(f"  Val RMSE: {train_history['val_metrics']['rmse']:.4f}")
    print(f"  Val R²:   {train_history['val_metrics']['r2']:.4f}")

    # Evaluate on test
    test_preds = model.predict(test_samples)
    test_labels = np.array([s.label for s in test_samples])
    test_metrics = compute_metrics(test_labels, test_preds)

    print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Test R²:   {test_metrics['r2']:.4f}")
    print(f"  Test Spearman: {test_metrics['spearman']:.4f}")

    return {
        'model_name': model_name,
        'val_metrics': train_history['val_metrics'],
        'test_metrics': test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on harder split")
    parser.add_argument(
        "--dataset",
        type=str,
        default="CycPeptMPDB_PAMPA",
        help="Dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="sequence_cluster",
        help="Split name"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/harder_split"),
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("HARDER GENERALIZATION SPLIT EVALUATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print()

    # Load data
    print("Loading data...")
    splits = load_pem_dataset(args.dataset, args.split)
    train_samples = splits['train']
    val_samples = splits['val']
    test_samples = splits['test']

    print(f"Train: {len(train_samples)} samples")
    print(f"Val: {len(val_samples)} samples")
    print(f"Test: {len(test_samples)} samples")

    # Train and evaluate models
    models = [
        "sequence_composition",
        "descriptor_only",
        "anchor_aware_descriptor",
    ]

    results = []

    for model_name in models:
        result = train_and_evaluate_model(
            model_name, train_samples, val_samples, test_samples, args.seed
        )
        results.append(result)

    # Save results as JSON
    json_file = args.output_dir / "harder_split_results.json"
    with open(json_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'split': args.split,
            'seed': args.seed,
            'results': results,
        }, f, indent=2)
    print(f"\nResults saved to {json_file}")

    # Save results as CSV
    csv_file = args.output_dir / "harder_split_results.csv"
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['model_name', 'test_rmse', 'test_mae', 'test_r2', 'test_spearman']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({
                'model_name': r['model_name'],
                'test_rmse': r['test_metrics']['rmse'],
                'test_mae': r['test_metrics']['mae'],
                'test_r2': r['test_metrics']['r2'],
                'test_spearman': r['test_metrics']['spearman'],
            })
    print(f"CSV saved to {csv_file}")

    # Print summary table
    print("\n" + "="*80)
    print("HARDER SPLIT EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Model':<30} | {'RMSE':>8} | {'MAE':>8} | {'R²':>8} | {'Spearman':>9}")
    print("-"*80)

    for r in results:
        print(f"{r['model_name']:<30} | {r['test_metrics']['rmse']:>8.4f} | "
              f"{r['test_metrics']['mae']:>8.4f} | {r['test_metrics']['r2']:>8.4f} | "
              f"{r['test_metrics']['spearman']:>9.4f}")

    print("="*80)

    # Compare with random split results if available
    random_results_file = Path("results/anchor_descriptor_xgb/test_results_aggregated.json")
    if random_results_file.exists():
        print("\nCOMPARISON WITH RANDOM SPLIT")
        print("="*80)

        with open(random_results_file, 'r') as f:
            random_results = json.load(f)

        # Get anchor-aware descriptor result from harder split
        harder_result = next(r for r in results if r['model_name'] == 'anchor_aware_descriptor')

        # Compare
        random_r2 = random_results['aggregated_metrics']['r2']['mean']
        random_spearman = random_results['aggregated_metrics']['spearman']['mean']

        harder_r2 = harder_result['test_metrics']['r2']
        harder_spearman = harder_result['test_metrics']['spearman']

        print(f"{'Split':<20} | {'R²':>8} | {'Spearman':>9}")
        print("-"*80)
        print(f"{'Random (easy)':<20} | {random_r2:>8.4f} | {random_spearman:>9.4f}")
        print(f"{'Sequence Cluster':<20} | {harder_r2:>8.4f} | {harder_spearman:>9.4f}")
        print(f"{'Degradation':<20} | {harder_r2 - random_r2:>+8.4f} | {harder_spearman - random_spearman:>+9.4f}")
        print("="*80)


if __name__ == "__main__":
    main()
