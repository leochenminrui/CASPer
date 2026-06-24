#!/usr/bin/env python3
"""
Generate publication-quality figures for manuscript.

Creates clean, Bioinformatics-style figures with consistent aesthetics.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
})

# Color palette
COLORS = {
    'composition': '#8DA0CB',  # Blue-gray
    'descriptor': '#FC8D62',   # Orange
    'anchor_aware': '#66C2A5',  # Teal
    'position': '#E78AC3',     # Pink
    'lm_embedding': '#A6D854', # Light green
}


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def figure_1_graded_perturbation():
    """
    Figure 1: Graded anchor perturbation curve.

    Shows progressive degradation of R² with increasing anchor position error.
    """
    print("\nGenerating Figure 1: Graded anchor perturbation...")

    # Load data
    data_file = Path("results/graded_perturbation/graded_perturbation_results.json")
    data = load_json(data_file)

    # Extract shift distances and R² values from results list
    results = sorted(data['results'], key=lambda x: x['shift_distance'])

    shifts = [r['shift_distance'] for r in results]
    r2_values = [r['r2'] for r in results]
    rmse_values = [r['rmse'] for r in results]
    spearman_values = [r['spearman'] for r in results]

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Panel A: R²
    axes[0].plot(shifts, r2_values, 'o-', color=COLORS['anchor_aware'], linewidth=2, markersize=6)
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Correct anchor')
    axes[0].set_xlabel('Anchor Shift Distance (residues)')
    axes[0].set_ylabel('R²')
    axes[0].set_title('A. R² vs Anchor Shift')
    axes[0].grid(True, alpha=0.3, linestyle=':')
    axes[0].legend()

    # Panel B: RMSE
    axes[1].plot(shifts, rmse_values, 'o-', color=COLORS['anchor_aware'], linewidth=2, markersize=6)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Correct anchor')
    axes[1].set_xlabel('Anchor Shift Distance (residues)')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('B. RMSE vs Anchor Shift')
    axes[1].grid(True, alpha=0.3, linestyle=':')
    axes[1].legend()

    # Panel C: Spearman ρ
    axes[2].plot(shifts, spearman_values, 'o-', color=COLORS['anchor_aware'], linewidth=2, markersize=6)
    axes[2].axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Correct anchor')
    axes[2].set_xlabel('Anchor Shift Distance (residues)')
    axes[2].set_ylabel('Spearman ρ')
    axes[2].set_title('C. Spearman ρ vs Anchor Shift')
    axes[2].grid(True, alpha=0.3, linestyle=':')
    axes[2].legend()

    plt.tight_layout()

    # Save
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    plt.savefig(output_dir / "Figure_1_graded_perturbation.pdf")
    plt.savefig(output_dir / "Figure_1_graded_perturbation.png")

    print(f"  Saved: {output_dir / 'Figure_1_graded_perturbation.pdf'}")

    # Save source data
    source_data = pd.DataFrame({
        'shift': shifts,
        'r2': r2_values,
        'rmse': rmse_values,
        'spearman_rho': spearman_values,
    })
    source_dir = output_dir / "source_data"
    source_dir.mkdir(exist_ok=True)
    source_data.to_csv(source_dir / "Figure_1_source_data.csv", index=False)

    plt.close()


def figure_2_main_comparison():
    """
    Figure 2: Main model comparison.

    Bar chart showing R² for all models with error bars.
    """
    print("\nGenerating Figure 2: Main model comparison...")

    # Load compiled metrics
    compiled_metrics = load_json("results/compiled_metrics.json")

    # Extract baseline data from compiled metrics
    baselines = compiled_metrics.get('baselines_random_split', {})
    composition_data = baselines.get('composition_xgboost', {})
    descriptor_data = baselines.get('descriptor_only_xgboost', {})

    # Load anchor-aware data (has different structure)
    anchor_data_raw = load_json("results/anchor_descriptor_xgb/test_results_aggregated.json")
    anchor_data = anchor_data_raw.get('aggregated_metrics', anchor_data_raw)

    # Check if LM embedding results exist
    lm_data_path = Path("results/baselines/CycPeptMPDB_PAMPA/lm_embedding_xgboost/test_results_aggregated.json")
    has_lm = lm_data_path.exists()

    if has_lm:
        lm_data = load_json(lm_data_path)

    # Prepare data
    models = ['Sequence\n(composition)', 'Descriptor-only', 'Anchor-aware\ndescriptor']
    r2_means = [
        composition_data['r2']['mean'],
        descriptor_data['r2']['mean'],
        anchor_data['r2']['mean'],
    ]
    r2_stds = [
        composition_data['r2']['std'],
        descriptor_data['r2']['std'],
        anchor_data['r2']['std'],
    ]
    colors = [COLORS['composition'], COLORS['descriptor'], COLORS['anchor_aware']]

    if has_lm:
        models.insert(1, 'Sequence\n(LM embedding)')
        r2_means.insert(1, lm_data['r2']['mean'])
        r2_stds.insert(1, lm_data['r2']['std'])
        colors.insert(1, COLORS['lm_embedding'])

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=5, color=colors,
                   edgecolor='black', linewidth=1.2, alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('R² (Coefficient of Determination)')
    ax.set_title('Model Performance on CycPeptMPDB-PAMPA (Random Split)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 0.6)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, r2_means, r2_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save
    output_dir = Path("figures")
    plt.savefig(output_dir / "Figure_2_main_comparison.pdf")
    plt.savefig(output_dir / "Figure_2_main_comparison.png")

    print(f"  Saved: {output_dir / 'Figure_2_main_comparison.pdf'}")

    # Save source data
    source_data = pd.DataFrame({
        'model': models,
        'r2_mean': r2_means,
        'r2_std': r2_stds,
    })
    source_data.to_csv(output_dir / "source_data" / "Figure_2_source_data.csv", index=False)

    plt.close()


def figure_3_feature_ablation():
    """
    Figure 3: Feature ablation analysis.

    Bar chart showing ΔR² for different feature groups.
    """
    print("\nGenerating Figure 3: Feature ablation...")

    # Load data
    data_file = Path("results/feature_ablation/ablation_results.json")
    data = load_json(data_file)

    # Results are stored in a list
    results_list = data['results']

    # Find baseline (chemistry only)
    baseline_result = next(r for r in results_list if r['ablation_mode'] == 'chemistry_only')
    baseline_r2 = baseline_result['test_metrics']['r2']

    # Prepare data
    variants = [
        'Position stats\n(A+B1)',
        'Residue comp.\n(A+B2)',
        'Local context\n(A+B3)',
        'All anchor\n(A+B)',
        'Attachment\n(A+C)',
        'Full model\n(A+B+C)',
    ]

    ablation_modes = [
        'chemistry_position',
        'chemistry_residue',
        'chemistry_context',
        'chemistry_anchors',
        'chemistry_attachment',
        'full',
    ]

    delta_r2 = []
    for mode in ablation_modes:
        result = next(r for r in results_list if r['ablation_mode'] == mode)
        delta_r2.append(result['test_metrics']['r2'] - baseline_r2)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(variants))
    colors_ablation = [COLORS['anchor_aware'] if d > 0 else '#E78AC3' for d in delta_r2]

    bars = ax.bar(x, delta_r2, color=colors_ablation, edgecolor='black',
                   linewidth=1.2, alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Feature Ablation Variant')
    ax.set_ylabel('ΔR² (vs Chemistry-Only Baseline)')
    ax.set_title('Contribution of Anchor-Aware Feature Groups')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=0)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')

    # Add value labels
    for bar, delta in zip(bars, delta_r2):
        height = bar.get_height()
        y_pos = height + 0.002 if height > 0 else height - 0.005
        va = 'bottom' if height > 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{delta:+.4f}',
                ha='center', va=va, fontsize=8)

    plt.tight_layout()

    # Save
    output_dir = Path("figures")
    plt.savefig(output_dir / "Figure_3_feature_ablation.pdf")
    plt.savefig(output_dir / "Figure_3_feature_ablation.png")

    print(f"  Saved: {output_dir / 'Figure_3_feature_ablation.pdf'}")

    # Save source data
    source_data = pd.DataFrame({
        'variant': variants,
        'delta_r2': delta_r2,
        'r2': [next(r for r in results_list if r['ablation_mode'] == mode)['test_metrics']['r2'] for mode in ablation_modes],
    })
    source_data.to_csv(output_dir / "source_data" / "Figure_3_source_data.csv", index=False)

    plt.close()


def figure_4_ood_generalization():
    """
    Figure 4: OOD generalization comparison.

    Shows performance degradation on stricter split.
    """
    print("\nGenerating Figure 4: OOD generalization...")

    # Load random split data from compiled metrics
    compiled_metrics = load_json("results/compiled_metrics.json")
    baselines = compiled_metrics.get('baselines_random_split', {})

    comp_random = baselines.get('composition_xgboost', {})
    desc_random = baselines.get('descriptor_only_xgboost', {})
    anchor_random_raw = load_json("results/anchor_descriptor_xgb/test_results_aggregated.json")
    anchor_random = anchor_random_raw.get('aggregated_metrics', anchor_random_raw)

    # Load harder split data
    harder_data = load_json("results/harder_split/harder_split_results.json")

    # Extract results from list
    harder_results = harder_data['results']

    # Prepare data
    models = ['Composition', 'Descriptor-only', 'Anchor-aware']
    random_r2 = [
        comp_random['r2']['mean'],
        desc_random['r2']['mean'],
        anchor_random['r2']['mean'],
    ]

    # Find results by model name
    comp_harder = next(r for r in harder_results if r['model_name'] == 'sequence_composition')
    desc_harder = next(r for r in harder_results if r['model_name'] == 'descriptor_only')
    anchor_harder = next(r for r in harder_results if r['model_name'] == 'anchor_aware_descriptor')

    harder_r2 = [
        comp_harder['test_metrics']['r2'],
        desc_harder['test_metrics']['r2'],
        anchor_harder['test_metrics']['r2'],
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, random_r2, width, label='Random split (70/15/15)',
                   color=COLORS['anchor_aware'], edgecolor='black', linewidth=1.2, alpha=0.8)
    bars2 = ax.bar(x + width/2, harder_r2, width, label='70% identity cluster split',
                   color=COLORS['descriptor'], edgecolor='black', linewidth=1.2, alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('R²')
    ax.set_title('Generalization to Novel Sequence Scaffolds')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 0.55)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save
    output_dir = Path("figures")
    plt.savefig(output_dir / "Figure_4_ood_generalization.pdf")
    plt.savefig(output_dir / "Figure_4_ood_generalization.png")

    print(f"  Saved: {output_dir / 'Figure_4_ood_generalization.pdf'}")

    # Save source data
    source_data = pd.DataFrame({
        'model': models,
        'random_split_r2': random_r2,
        'harder_split_r2': harder_r2,
        'degradation': [r - h for r, h in zip(random_r2, harder_r2)],
    })
    source_data.to_csv(output_dir / "source_data" / "Figure_4_source_data.csv", index=False)

    plt.close()


def generate_figure_legends():
    """Generate figure legends file."""
    print("\nGenerating figure legends...")

    legends = """# Figure Legends

## Figure 1: Graded Anchor Perturbation Reveals Position Sensitivity

Performance degradation of the anchor-aware descriptor model as a function of systematic anchor position perturbation. Anchor positions were shifted by {-5, -3, -2, -1, 0, +1, +2, +3, +5} residues relative to the correct positions. **(A)** R² (coefficient of determination) drops sharply with even single-residue errors, falling from 0.47 (correct) to ~0.29 (±1 residue). **(B)** RMSE (root mean squared error) increases monotonically with shift distance. **(C)** Spearman rank correlation degrades progressively but remains positive even at ±5 residues, indicating that approximate positional information retains some value. The red dashed line indicates correct anchor positions (shift = 0). Gray dashed line in panel A shows R² = 0 (trivial baseline).

## Figure 2: Anchor-Aware Descriptor Model Outperforms Sequence and Chemistry Baselines

Performance comparison of baseline models on the CycPeptMPDB-PAMPA random split (70/15/15 train/val/test). Error bars show standard deviation across 5 random seeds (composition and descriptor-only) or 3 seeds (anchor-aware). Chemistry descriptors alone (Descriptor-only, R² = 0.43 ± 0.00) substantially outperform amino acid composition (R² = 0.26 ± 0.00), demonstrating that chemical modification properties contain the dominant predictive signal. Pretrained protein language model embeddings (Sequence LM embedding) provide a stronger sequence baseline than composition. Anchor-aware descriptors (R² = 0.47 ± 0.00) add measurable value (+0.04 R² improvement) by incorporating precise modification site information.

## Figure 3: Position Statistics Are the Primary Anchor-Aware Feature Component

Contribution of anchor-aware feature groups to prediction performance, measured as ΔR² relative to chemistry-only baseline (Group A: 10 global chemical descriptors). Group B features aggregate properties of residues at anchor positions: B1 = position statistics (count, density, mean, std, span), B2 = residue composition (per-AA frequencies), B3 = local context (hydrophobic/charged/polar fractions). Group C captures attachment-aware multi-edit features. Position statistics (B1) provide the largest individual contribution (+0.012 ΔR²). Residue composition (B2) and local context (B3) are weak or negative alone but contribute synergistically when combined with B1. Full model (A+B+C) achieves +0.040 ΔR² improvement over chemistry-only baseline.

## Figure 4: Chemistry Descriptors Generalize to Novel Scaffolds, Anchor Gains Diminish

Performance degradation under stricter sequence-cluster split (70% identity threshold) that ensures test set contains novel sequence scaffolds not seen during training. Composition baseline fails catastrophically (R² drops from 0.26 to 0.04), confirming that amino acid composition alone does not generalize. Chemistry descriptors (Descriptor-only) retain substantial predictive power (R² = 0.27 on cluster split vs 0.43 on random split, -37% degradation), demonstrating that chemical modification properties provide the primary transferable signal. Anchor-aware features add minimal value on novel scaffolds (ΔR² = +0.002 vs +0.039 on random split), suggesting that anchor-site feature engineering is partially sequence-context-dependent.
"""

    output_dir = Path("figures")
    with open(output_dir / "FIGURE_LEGENDS.md", 'w') as f:
        f.write(legends)

    print(f"  Saved: {output_dir / 'FIGURE_LEGENDS.md'}")


def main():
    """Generate all publication figures."""
    print("="*80)
    print("Generating Publication-Quality Figures")
    print("="*80)

    # Create output directories
    Path("figures/source_data").mkdir(parents=True, exist_ok=True)

    # Generate figures
    figure_1_graded_perturbation()
    figure_2_main_comparison()
    figure_3_feature_ablation()
    figure_4_ood_generalization()
    generate_figure_legends()

    print("\n" + "="*80)
    print("All figures generated successfully!")
    print("="*80)
    print(f"\nOutput directory: figures/")
    print(f"  - Figure_1_graded_perturbation.pdf")
    print(f"  - Figure_2_main_comparison.pdf")
    print(f"  - Figure_3_feature_ablation.pdf")
    print(f"  - Figure_4_ood_generalization.pdf")
    print(f"  - FIGURE_LEGENDS.md")
    print(f"  - source_data/*.csv")


if __name__ == "__main__":
    main()
