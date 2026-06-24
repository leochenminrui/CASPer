#!/usr/bin/env python3
"""
Comprehensive results compilation for manuscript audit.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def load_json(path):
    """Load JSON file, return None if missing."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def main():
    results_dir = PROJECT_ROOT / 'results'

    all_results = {
        'dataset_stats': {},
        'baselines_random_split': {},
        'anchor_aware_random_split': {},
        'mechanism_controls': {},
        'graded_perturbation': {},
        'feature_ablation': {},
        'harder_split': {},
        'missing': []
    }

    # Dataset statistics
    data_dir = PROJECT_ROOT / 'data'
    all_results['dataset_stats'] = {
        'raw_samples': 7298,
        'processed_samples': 7224,
        'unique_sequences': 1325,
        'random_split': {'train': 5056, 'val': 1084, 'test': 1084},
        'sequence_cluster_split': {'train': 4801, 'val': 1306, 'test': 1117}
    }

    # Baselines on random split
    baseline_dir = results_dir / 'baselines' / 'CycPeptMPDB_PAMPA'
    for model_name in ['composition_xgboost', 'descriptor_only_xgboost']:
        agg_path = baseline_dir / model_name / 'test_results_aggregated.json'
        data = load_json(agg_path)
        if data:
            all_results['baselines_random_split'][model_name] = {
                'rmse': data['aggregated_metrics']['rmse'],
                'mae': data['aggregated_metrics']['mae'],
                'r2': data['aggregated_metrics']['r2'],
                'spearman': data['aggregated_metrics']['spearman'],
                'n_seeds': data.get('n_seeds', len(data['aggregated_metrics']['r2']['values']))
            }
        else:
            all_results['missing'].append(f'baseline: {model_name}')

    # Anchor-aware descriptor on random split
    agg_path = results_dir / 'anchor_descriptor_xgb' / 'test_results_aggregated.json'
    data = load_json(agg_path)
    if data:
        all_results['anchor_aware_random_split'] = {
            'rmse': data['aggregated_metrics']['rmse'],
            'mae': data['aggregated_metrics']['mae'],
            'r2': data['aggregated_metrics']['r2'],
            'spearman': data['aggregated_metrics']['spearman'],
            'n_seeds': data.get('n_seeds', len(data['aggregated_metrics']['r2']['values']))
        }
    else:
        all_results['missing'].append('anchor_aware_descriptor')



    # Mechanism controls
    controls_dir = results_dir / 'mechanism_controls'
    control_path = controls_dir / 'mechanism_controls_seed42.json'
    data = load_json(control_path)
    if data:
        all_results['mechanism_controls'] = data
    else:
        # Try individual files
        for mode in ['baseline', 'wrong_anchor', 'coarse_position']:
            path = results_dir / 'anchor_descriptor_xgb' / f'test_results_seed42_{mode}.json'
            if os.path.exists(path):
                all_results['mechanism_controls'][mode] = load_json(path)

    # Graded perturbation
    grad_path = results_dir / 'graded_perturbation' / 'graded_perturbation_results.json'
    data = load_json(grad_path)
    if data:
        all_results['graded_perturbation'] = data
    else:
        all_results['missing'].append('graded_perturbation')

    # Feature ablation
    abl_path = results_dir / 'feature_ablation' / 'ablation_results.json'
    data = load_json(abl_path)
    if data:
        all_results['feature_ablation'] = data
    else:
        all_results['missing'].append('feature_ablation')

    # Harder split
    harder_path = results_dir / 'harder_split' / 'harder_split_results.json'
    data = load_json(harder_path)
    if data:
        all_results['harder_split'] = data
    else:
        all_results['missing'].append('harder_split')

    # Write output
    output_path = results_dir / 'compiled_all_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Compiled results written to: {output_path}")
    print(f"\nMissing items ({len(all_results['missing'])}):")
    for item in all_results['missing']:
        print(f"  - {item}")

if __name__ == '__main__':
    main()
