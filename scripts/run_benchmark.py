#!/usr/bin/env python3
"""
CASPer Unified Benchmark Runner.

Fair comparison of all models on identical data, splits, and metrics.
Supports: quick mode (10 trials, 3 seeds), full mode (50 trials, 5 seeds).

Usage:
    # Quick benchmark (development / sanity check)
    python scripts/run_benchmark.py --quick

    # Full benchmark (manuscript quality)
    python scripts/run_benchmark.py --full

    # Custom
    python scripts/run_benchmark.py --config configs/benchmark/benchmark_full.yaml
    python scripts/run_benchmark.py --split random --models full_ABC_xgb chem_A_xgb --n-trials 30 --seeds 0 1 2
    python scripts/run_benchmark.py --full --resume

Output:
    results/benchmark/
        random_split/seed_0/{model_id}/metrics.json, predictions.csv, best_params.json, optuna_trials.csv
        ...
        summary/{tables and plots}
"""

import argparse
import sys
import logging
from pathlib import Path
import yaml
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.registry import MODEL_REGISTRY, list_implemented_models, list_models_by_role
from src.benchmark.runner import BenchmarkRunner, run_single_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def list_models():
    """Print all registered models with their status."""
    print(f"\n{'='*90}")
    print("CASPer Benchmark — Model Registry")
    print(f"{'='*90}")
    print(f"{'Model ID':<30} {'Role':<22} {'Status':<18} {'HPO':<6}")
    print("-" * 90)
    for mid, spec in MODEL_REGISTRY.items():
        hpo = "Yes" if spec.hpo else f"No ({spec.hpo_reason[:30]})"
        print(f"{mid:<30} {spec.role:<22} {spec.status:<18} {hpo:<6}")
    print("-" * 90)
    print(f"\nTotal: {len(MODEL_REGISTRY)} models "
          f"({len(list_implemented_models())} implemented, "
          f"{len(MODEL_REGISTRY) - len(list_implemented_models())} require external deps)")
    print()


def generate_summary_tables(output_base: Path):
    """Post-hoc: generate all summary tables from completed results."""
    logger.info("Generating summary tables...")
    summary_dir = output_base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Collect all metrics.json files
    all_results = []
    for metrics_file in sorted(output_base.rglob("metrics.json")):
        try:
            with open(metrics_file) as f:
                data = json.load(f)
            if data.get('status') in ('completed', 'not_run', 'failed'):
                all_results.append(data)
        except Exception:
            pass

    if not all_results:
        logger.warning("No results found to summarize.")
        return

    # ── all_metrics_long.csv ─────────────────────────────────────────────
    import csv
    long_file = summary_dir / "all_metrics_long.csv"
    metrics_keys = ['rmse', 'mae', 'r2', 'spearman', 'pearson',
                    'kendall_tau', 'variance_ratio', 'n_samples']
    with open(long_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model_id', 'role', 'seed', 'split_type', 'status', 'hpo',
            'n_train', 'n_val', 'n_test', 'n_features',
            'metric', 'value'
        ])
        for r in all_results:
            base = [r.get('model_id', ''), r.get('role', ''),
                    r.get('seed', ''), r.get('split_type', ''),
                    r.get('status', ''), r.get('hpo', ''),
                    r.get('n_train', ''), r.get('n_val', ''),
                    r.get('n_test', ''), r.get('n_features', '')]
            test_m = r.get('test_metrics', {})
            if test_m:
                for mk in metrics_keys:
                    if mk in test_m:
                        writer.writerow(base + [mk, test_m[mk]])
    logger.info(f"  {long_file} ({len(all_results)} results)")

    # ── all_metrics_wide.csv ─────────────────────────────────────────────
    wide_file = summary_dir / "all_metrics_wide.csv"
    # Group by model_id + split_type + seed
    with open(wide_file, 'w', newline='') as f:
        fieldnames = ['model_id', 'role', 'seed', 'split_type', 'status',
                      'hpo', 'n_train', 'n_val', 'n_test'] + metrics_keys
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in ['model_id', 'role', 'seed',
                   'split_type', 'status', 'hpo', 'n_train', 'n_val', 'n_test']}
            test_m = r.get('test_metrics', {})
            for mk in metrics_keys:
                row[mk] = test_m.get(mk, '')
            writer.writerow(row)
    logger.info(f"  {wide_file}")

    # ── mean_std_by_model.csv ────────────────────────────────────────────
    ms_file = summary_dir / "mean_std_by_model.csv"
    groups = {}
    for r in all_results:
        if r.get('status') != 'completed':
            continue
        key = (r.get('model_id', ''), r.get('split_type', ''))
        if key not in groups:
            groups[key] = {mk: [] for mk in metrics_keys}
        test_m = r.get('test_metrics', {})
        for mk in metrics_keys:
            if mk in test_m and np.isfinite(test_m[mk]):
                groups[key][mk].append(test_m[mk])

    with open(ms_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model_id', 'split_type', 'n_seeds',
            'rmse_mean', 'rmse_std', 'mae_mean', 'mae_std',
            'r2_mean', 'r2_std', 'spearman_mean', 'spearman_std',
            'pearson_mean', 'pearson_std',
        ])
        for (mid, st), vals in sorted(groups.items()):
            row = [mid, st, len(vals.get('r2', []))]
            for mk in ['rmse', 'mae', 'r2', 'spearman', 'pearson']:
                arr = np.array(vals.get(mk, []))
                row.extend([float(np.mean(arr)), float(np.std(arr, ddof=1))
                           if len(arr) > 1 else 0.0])
            writer.writerow(row)
    logger.info(f"  {ms_file}")

    # ── model_family_table.csv ───────────────────────────────────────────
    mf_file = summary_dir / "model_family_table.csv"
    with open(mf_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'role_in_study', 'model_id', 'model_name', 'representation',
            'purpose', 'hpo', 'status'
        ])
        for mid, spec in MODEL_REGISTRY.items():
            writer.writerow([
                spec.role, mid, spec.model_name, spec.featurizer,
                spec.purpose, str(spec.hpo), spec.status,
            ])
    logger.info(f"  {mf_file}")

    # ── benchmark_table_for_manuscript.csv ───────────────────────────────
    bm_file = summary_dir / "benchmark_table_for_manuscript.csv"
    with open(bm_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model', 'role', 'split', 'RMSE_mean', 'RMSE_SD',
            'MAE_mean', 'MAE_SD', 'R2_mean', 'R2_SD',
            'Spearman_mean', 'Spearman_SD', 'n_seeds', 'hpo'
        ])
        for (mid, st), vals in sorted(groups.items()):
            spec = MODEL_REGISTRY.get(mid, None)
            role = spec.role if spec else ''
            hpo = spec.hpo if spec else ''
            n = len(vals.get('r2', []))
            row = [mid, role, st]
            for mk in ['rmse', 'mae', 'r2', 'spearman']:
                arr = np.array(vals.get(mk, []))
                row.append(f"{np.mean(arr):.4f}")
                row.append(f"{np.std(arr, ddof=1):.4f}" if n > 1 else "—")
            row.extend([n, hpo])
            writer.writerow(row)
    logger.info(f"  {bm_file}")

    # ── external_benchmark_status.csv ────────────────────────────────────
    ext_file = summary_dir / "external_benchmark_status.csv"
    with open(ext_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_id', 'status', 'reason'])
        for r in all_results:
            if r.get('status') in ('not_run', 'failed'):
                writer.writerow([
                    r.get('model_id', ''),
                    r.get('status', ''),
                    r.get('reason', r.get('error', '')),
                ])
    logger.info(f"  {ext_file}")

    # ── ablation_table.csv ───────────────────────────────────────────────
    ablation_models = [
        'chem_A_xgb', 'site_B_xgb', 'context_C_xgb', 'site_context_BC_xgb',
        'chem_B1_xgb', 'chem_B2_xgb', 'chem_B3_xgb',
        'chem_site_AB_xgb', 'chem_context_AC_xgb', 'full_ABC_xgb',
    ]
    abl_file = summary_dir / "ablation_table.csv"
    with open(abl_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model', 'group', 'split', 'R2_mean', 'R2_SD',
            'Spearman_mean', 'Spearman_SD', 'n_seeds'
        ])
        for mid in ablation_models:
            for st in ['random']:  # primary split
                vals = groups.get((mid, st))
                if vals:
                    r2_arr = np.array(vals.get('r2', []))
                    sp_arr = np.array(vals.get('spearman', []))
                    n = len(r2_arr)
                    writer.writerow([
                        mid, 'see_legend', st,
                        f"{np.mean(r2_arr):.4f}",
                        f"{np.std(r2_arr, ddof=1):.4f}" if n > 1 else "—",
                        f"{np.mean(sp_arr):.4f}",
                        f"{np.std(sp_arr, ddof=1):.4f}" if n > 1 else "—",
                        n,
                    ])
    logger.info(f"  {abl_file}")

    logger.info(f"\nAll summary tables saved to: {summary_dir}")


def generate_benchmark_figures(output_base: Path):
    """Generate benchmark figures from summary data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import json, csv
    from collections import defaultdict

    summary_dir = output_base / "summary"
    ms_file = summary_dir / "mean_std_by_model.csv"
    if not ms_file.exists():
        logger.warning("No mean_std_by_model.csv — run summary first")
        return

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
        "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    })

    # Parse mean_std_by_model.csv
    data = defaultdict(dict)
    with open(ms_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['model_id'], row['split_type'])
            data[key] = row

    def make_barplot(metric_key, metric_label, filename):
        """Generic bar plot by model role."""
        role_order = ['internal_baseline', 'primary_paper', 'ablation_control',
                      'generic_benchmark', 'auxiliary']
        fig, ax = plt.subplots(figsize=(legacy_max(12, len(data) * 0.3), 5))
        models_ordered = []
        means = []
        stds = []
        colors = []
        color_map = {
            'internal_baseline': '#8DA0CB',
            'primary_paper': '#66C2A5',
            'ablation_control': '#FC8D62',
            'generic_benchmark': '#E78AC3',
            'auxiliary': '#A6D854',
        }
        for (mid, st), row in sorted(data.items()):
            if st != 'random':
                continue
            spec = MODEL_REGISTRY.get(mid)
            if spec is None:
                continue
            mean_col = f'{metric_key}_mean'
            std_col = f'{metric_key}_std'
            if mean_col not in row:
                continue
            models_ordered.append(mid)
            means.append(float(row[mean_col]))
            stds.append(float(row[std_col]))
            colors.append(color_map.get(spec.role, '#CCCCCC'))

        x = np.arange(len(models_ordered))
        ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor='black',
               linewidth=0.8, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models_ordered, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(metric_label)
        ax.set_title(f'Benchmark {metric_label} — Random Split (mean ± SD across seeds)')
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        plt.tight_layout()
        fig.savefig(summary_dir / filename)
        plt.close(fig)
        logger.info(f"  Saved {summary_dir / filename}")

    make_barplot('r2', 'Test R²', 'benchmark_R2_barplot.pdf')
    make_barplot('spearman', 'Test Spearman ρ', 'benchmark_spearman_barplot.pdf')

    # Ablation bar plot
    ablation_order = [
        'chem_A_xgb', 'site_B_xgb', 'context_C_xgb', 'site_context_BC_xgb',
        'chem_B1_xgb', 'chem_B2_xgb', 'chem_B3_xgb',
        'chem_site_AB_xgb', 'chem_context_AC_xgb', 'full_ABC_xgb',
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(ablation_order))
    r2_means_abl = []
    r2_stds_abl = []
    labels_abl = []
    for mid in ablation_order:
        row = data.get((mid, 'random'), {})
        if not row:
            r2_means_abl.append(0)
            r2_stds_abl.append(0)
        else:
            r2_means_abl.append(float(row.get('r2_mean', 0)))
            r2_stds_abl.append(float(row.get('r2_std', 0)))
        labels_abl.append(mid.replace('_xgb', '').replace('_', ' '))

    colors_abl = ['#8DA0CB'] + ['#FC8D62'] * 3 + ['#E78AC3'] * 3 + \
                 ['#66C2A5', '#A6D854'] + ['#FFD700']
    ax.bar(x, r2_means_abl, yerr=r2_stds_abl, capsize=3,
           color=colors_abl[:len(ablation_order)], edgecolor='black',
           linewidth=0.8, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_abl, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Test R²')
    ax.set_title('Feature Group Ablation — Random Split (mean ± SD)')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    plt.tight_layout()
    fig.savefig(summary_dir / 'ablation_R2_barplot.pdf')
    plt.close(fig)
    logger.info(f"  Saved {summary_dir / 'ablation_R2_barplot.pdf'}")


def legacy_max(a, b):
    return a if a > b else b


def main():
    parser = argparse.ArgumentParser(
        description="CASPer Unified Benchmark Runner"
    )
    # Mode flags
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10 trials, 3 seeds, random split only")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: 50 trials, 5 seeds, both splits")
    parser.add_argument("--list", action="store_true",
                        help="List all registered models and exit")

    # Custom overrides
    parser.add_argument("--config", type=Path,
                        help="Path to benchmark YAML config")
    parser.add_argument("--split", type=str, nargs='+',
                        choices=['random', 'sequence_cluster'],
                        help="Split type(s) to use")
    parser.add_argument("--models", type=str, nargs='+',
                        help="Model IDs to run (default: all implemented)")
    parser.add_argument("--seeds", type=int, nargs='+',
                        help="Random seeds")
    parser.add_argument("--n-trials", type=int,
                        help="Optuna trials per model")

    # Behavior flags
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip completed runs (default)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Force re-run all models")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only generate summary tables from existing results")
    parser.add_argument("--figures-only", action="store_true",
                        help="Only generate figures from existing summary")

    args = parser.parse_args()

    # List mode
    if args.list:
        list_models()
        return

    # Determine config
    if args.quick:
        config_path = PROJECT_ROOT / "configs/benchmark/benchmark_quick.yaml"
    elif args.full:
        config_path = PROJECT_ROOT / "configs/benchmark/benchmark_full.yaml"
    elif args.config:
        config_path = args.config
    else:
        # Default to quick
        config_path = PROJECT_ROOT / "configs/benchmark/benchmark_quick.yaml"
        logger.info("No mode specified, using --quick by default")

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Override config with CLI args
    if args.split:
        config.setdefault('benchmark', {})['splits'] = args.split
    if args.seeds:
        config.setdefault('benchmark', {})['seeds'] = args.seeds
    if args.n_trials:
        config.setdefault('benchmark', {})['n_trials'] = args.n_trials

    output_base = Path(config.get('output', {}).get(
        'base_dir', 'results/benchmark'))

    # Summary-only mode
    if args.summary_only:
        generate_summary_tables(output_base)
        return

    # Figures-only mode
    if args.figures_only:
        generate_benchmark_figures(output_base)
        return

    # Determine models
    models = args.models
    if models is None:
        include = config.get('benchmark', {}).get('models', {}).get('include', [])
        exclude = set(config.get('benchmark', {}).get('models', {}).get(
            'exclude', []))
        if include:
            models = [m for m in include if m not in exclude]
        else:
            models = list_implemented_models()

    # Run benchmark
    runner = BenchmarkRunner(config)
    runner.run(
        models=models,
        splits=args.split,
        seeds=args.seeds,
        resume=not args.no_resume,
    )

    # Auto-generate summary
    generate_summary_tables(output_base)
    generate_benchmark_figures(output_base)

    logger.info("\nDone! Results in: " + str(output_base))
    logger.info("Summary tables in: " + str(output_base / "summary"))
    logger.info("Run with --summary-only to regenerate tables from existing results.")


if __name__ == "__main__":
    import numpy as np
    main()
