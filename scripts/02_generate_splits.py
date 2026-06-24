#!/usr/bin/env python3
"""
Generate data splits for all datasets.

Applies all feasible split strategies and generates:
- Split JSONL files (train/val/test)
- Split metadata JSON
- Split summary reports

Usage:
    python scripts/02_generate_splits.py
    python scripts/02_generate_splits.py --dataset CycPeptMPDB_PAMPA
    python scripts/02_generate_splits.py --strategies random scaffold_aware
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pem_schema import PEMSample
from src.data.serialization import load_samples_jsonl, save_samples_jsonl
from src.data.splitting import (
    RandomSplitter,
    ScaffoldAwareSplitter,
    SequenceClusterSplitter,
    EditFamilyAwareSplitter,
    SameEditFamilyNewScaffoldSplitter,
    GroupedDerivativeSplitter,
    SplitResult,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset configurations — single dataset for this study
DATASET_CONFIGS = {
    'CycPeptMPDB_PAMPA': {
        'input_file': 'data/processed/pem_schema/cycpeptmpdb_pampa.jsonl',
        'priority_strategies': [
            'scaffold_aware',
            'same_edit_family_new_scaffold',
            'edit_family_aware',
            'sequence_cluster',
            'random',
        ],
    },
}


# Strategy factory
def create_splitter(strategy_name: str, **kwargs):
    """Create a splitter instance by name."""
    strategies = {
        'random': RandomSplitter,
        'scaffold_aware': ScaffoldAwareSplitter,
        'sequence_cluster': SequenceClusterSplitter,
        'edit_family_aware': EditFamilyAwareSplitter,
        'same_edit_family_new_scaffold': SameEditFamilyNewScaffoldSplitter,
        'grouped_derivative': GroupedDerivativeSplitter,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategies[strategy_name](**kwargs)


def save_split_result(
    result: SplitResult,
    dataset_name: str,
    output_dir: Path,
) -> None:
    """
    Save split result to files.

    Creates:
    - {strategy}/train.jsonl
    - {strategy}/val.jsonl
    - {strategy}/test.jsonl
    - {strategy}/metadata.json
    """
    strategy_dir = output_dir / dataset_name / result.metadata.strategy_name
    strategy_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    save_samples_jsonl(result.train_samples, strategy_dir / 'train.jsonl')
    save_samples_jsonl(result.val_samples, strategy_dir / 'val.jsonl')
    save_samples_jsonl(result.test_samples, strategy_dir / 'test.jsonl')

    # Save metadata
    metadata_file = strategy_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(result.metadata.to_dict(), f, indent=2)

    logger.info(f"  Saved {result.metadata.strategy_name} split:")
    logger.info(f"    Train: {len(result.train_samples)}")
    logger.info(f"    Val:   {len(result.val_samples)}")
    logger.info(f"    Test:  {len(result.test_samples)}")
    logger.info(f"    Leakage risk: {result.metadata.leakage_analysis.overall_risk}")


def generate_splits_for_dataset(
    dataset_name: str,
    strategies: List[str] = None,
    output_dir: Path = Path('data/splits'),
) -> Dict[str, Any]:
    """
    Generate splits for a dataset.

    Args:
        dataset_name: Name of dataset
        strategies: List of strategy names (None = use all priority strategies)
        output_dir: Output directory for splits

    Returns:
        Summary dict with results for each strategy
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Generating splits for {dataset_name}")
    logger.info(f"{'=' * 60}")

    # Get config
    if dataset_name not in DATASET_CONFIGS:
        logger.error(f"Unknown dataset: {dataset_name}")
        return {}

    config = DATASET_CONFIGS[dataset_name]

    # Load samples
    input_file = Path(config['input_file'])
    if not input_file.exists():
        logger.warning(f"Input file not found: {input_file}")
        logger.warning(f"Skipping {dataset_name}")
        return {}

    logger.info(f"Loading samples from {input_file}")
    samples = load_samples_jsonl(input_file)
    logger.info(f"Loaded {len(samples)} samples")

    # Determine strategies to try
    if strategies is None:
        strategies = config['priority_strategies']

    # Try each strategy
    results = {}
    successful_strategies = []

    for strategy_name in strategies:
        logger.info(f"\n--- Strategy: {strategy_name} ---")

        try:
            # Create splitter
            splitter = create_splitter(strategy_name)

            # Check feasibility
            is_feasible, reason = splitter.check_feasibility(samples)

            if not is_feasible:
                logger.warning(f"Strategy not feasible: {reason}")
                results[strategy_name] = {
                    'feasible': False,
                    'reason': reason,
                }
                continue

            # Generate split
            logger.info("Generating split...")
            result = splitter.split(samples)

            # Save result
            save_split_result(result, dataset_name, output_dir)

            # Record success
            successful_strategies.append(strategy_name)
            results[strategy_name] = {
                'feasible': True,
                'train_count': len(result.train_samples),
                'val_count': len(result.val_samples),
                'test_count': len(result.test_samples),
                'leakage_risk': result.metadata.leakage_analysis.overall_risk,
            }

        except Exception as e:
            logger.error(f"Error applying strategy {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
            results[strategy_name] = {
                'feasible': False,
                'reason': f"Error: {str(e)}",
            }

    # Save overall metadata
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        'dataset': dataset_name,
        'total_samples': len(samples),
        'strategies': results,
        'successful_strategies': successful_strategies,
        'recommended_strategy': successful_strategies[0] if successful_strategies else None,
    }

    metadata_file = dataset_dir / 'split_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✓ Successfully applied {len(successful_strategies)} strategies")
    logger.info(f"✓ Saved metadata to {metadata_file}")

    return metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate data splits for PEM datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(DATASET_CONFIGS.keys()) + ['all'],
        default='all',
        help='Dataset to split (default: all)'
    )
    parser.add_argument(
        '--strategies',
        type=str,
        nargs='+',
        default=None,
        help='Strategies to apply (default: use priority list)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/splits'),
        help='Output directory (default: data/splits)'
    )

    args = parser.parse_args()

    # Determine datasets
    if args.dataset == 'all':
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = [args.dataset]

    # Generate splits
    all_results = {}
    for dataset_name in datasets:
        result = generate_splits_for_dataset(
            dataset_name,
            strategies=args.strategies,
            output_dir=args.output_dir,
        )
        all_results[dataset_name] = result

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")

    for dataset_name, result in all_results.items():
        if not result:
            logger.info(f"{dataset_name}: SKIPPED (no data)")
            continue

        n_success = len(result.get('successful_strategies', []))
        recommended = result.get('recommended_strategy', 'None')
        logger.info(f"{dataset_name}: {n_success} strategies, recommended: {recommended}")

    logger.info(f"\n✓ Done! Splits saved to {args.output_dir}")


if __name__ == '__main__':
    main()
