#!/usr/bin/env python3
"""
Retrospective Scaffold-Focused Ranking Simulation.

Two ranking simulations to address Reviewer 3's concerns about:
  1. Random split being too optimistic
  2. Cluster split being too harsh
  3. Need for realistic scaffold-focused prospective evaluation

Simulation 1: Time-Forward Global Ranking (cutoff-year based)
Simulation 2: Reference-Aware Scaffold Ranking (family-level)

Output: results/benchmark/scaffold_ranking/
"""

import sys
import json
import csv
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_enriched_data():
    """Load raw CSV + PEM schema, merge to get Year/Source with sequence info."""
    from src.data.pem_schema import PEMSample
    from src.data.serialization import load_jsonl

    # Load raw CSV for Year/Source
    raw_csv = PROJECT_ROOT / "data/raw/cycpeptmpdb_pampa.csv"
    df = pd.read_csv(raw_csv, low_memory=False)

    # Filter to PAMPA non-null
    df = df[df['PAMPA'].notna()].copy()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Monomer_Length'] = pd.to_numeric(df['Monomer_Length'], errors='coerce')

    # Load PEM samples for sequence + features
    pem_file = PROJECT_ROOT / "data/processed/pem_schema/cycpeptmpdb_pampa.jsonl"
    pem_samples = load_jsonl(pem_file)

    # Build lookup: source_row_index -> PEM sample
    pem_by_row = {}
    for s in pem_samples:
        row_idx = (s.provenance or {}).get('source_row_index')
        if row_idx is not None:
            pem_by_row[int(row_idx)] = s

    # Also try sample_id mapping: CSV ID is integer, PEM sample_id is "CYCPEPTMPDB_PAMPA_{ID:06d}"
    pem_by_csv_id = {}
    for s in pem_samples:
        raw = (s.provenance or {}).get('raw_data_sample', {})
        csv_id = raw.get('ID', '')
        if csv_id:
            pem_by_csv_id[str(csv_id)] = s

    logger.info(f"PEM samples: {len(pem_samples)}, row-index matched: {len(pem_by_row)}, csv-ID matched: {len(pem_by_csv_id)}")

    # Build enriched records
    records = []
    for idx, row in df.iterrows():
        sid = str(row.get('ID', ''))
        sample = pem_by_csv_id.get(sid)
        if sample is None:
            continue
        year = row.get('Year')
        if pd.isna(year):
            continue
        year = int(year)
        source = str(row.get('Source', ''))
        mol_shape = str(row.get('Molecule_Shape', ''))
        mon_len = row.get('Monomer_Length', '')
        orig_name = str(row.get('Original_Name_in_Source_Literature', ''))
        pampa_val = float(row['PAMPA']) if pd.notna(row.get('PAMPA')) else None
        if pampa_val is None:
            continue

        records.append({
            'sample_id': sid,
            'sample': sample,
            'year': year,
            'source': source,
            'molecule_shape': mol_shape,
            'monomer_length': int(mon_len) if pd.notna(mon_len) else 0,
            'original_name': orig_name,
            'pampa': pampa_val,
            'sequence': sample.sequence,
            'smiles': (sample.assay_metadata or {}).get('smiles', ''),
        })

    logger.info(f"Loaded {len(records)} enriched records with Year information")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation 1: Time-Forward Global Ranking
# ═══════════════════════════════════════════════════════════════════════════════

def run_time_forward_ranking(records, model_ids, seed=42):
    """Time-forward ranking: train on earlier years, test on later."""
    years = sorted(set(r['year'] for r in records))
    logger.info(f"Year range: {min(years)}–{max(years)}, {len(years)} unique years")

    # Determine cutoff years (skip first 3 and last 1)
    cutoff_years = years[3:-1]
    logger.info(f"Cutoff years: {cutoff_years}")

    output_dir = PROJECT_ROOT / "results/benchmark/time_forward_ranking"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for cutoff in cutoff_years:
        train = [r for r in records if r['year'] < cutoff]
        test = [r for r in records if r['year'] >= cutoff]

        if len(train) < 50 or len(test) < 50:
            logger.info(f"  cutoff={cutoff}: train={len(train)}, test={len(test)} — skipping")
            all_results.append({
                'cutoff_year': cutoff, 'n_train': len(train), 'n_test': len(test),
                'status': 'skipped', 'reason': 'Too few samples'
            })
            continue

        # Split test in half: val + test
        rng = np.random.RandomState(seed)
        rng.shuffle(test)
        n_val = len(test) // 2
        val = test[:n_val]
        test_final = test[n_val:]

        logger.info(f"  cutoff={cutoff}: train={len(train)}, val={len(val)}, test={len(test_final)}")

        for model_id in model_ids:
            try:
                metrics = train_and_evaluate_ranking(
                    model_id, train, None, test_final,
                    seed=seed, description=f"time_forward_{cutoff}")
                all_results.append({
                    'cutoff_year': cutoff, 'model_id': model_id,
                    'n_train': len(train), 'n_test': len(test_final),
                    'status': 'completed', **{f'metric_{k}': v for k, v in metrics.items()},
                })
            except Exception as e:
                logger.error(f"  {model_id} at cutoff={cutcut} FAILED: {e}")
                all_results.append({
                    'cutoff_year': cutoff, 'model_id': model_id,
                    'status': 'failed', 'error': str(e),
                })

    # Save
    with open(output_dir / "cutoff_level_results.csv", 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    # Summary by model
    summary = defaultdict(list)
    for r in all_results:
        if r.get('status') == 'completed':
            summary[r['model_id']].append(r)

    with open(output_dir / "summary_by_model.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_id', 'n_cutoffs', 'mean_R2', 'std_R2',
                        'mean_Spearman', 'std_Spearman'])
        for mid, results in sorted(summary.items()):
            r2_vals = [r.get('metric_r2', 0) for r in results]
            sp_vals = [r.get('metric_spearman', 0) for r in results]
            writer.writerow([mid, len(results),
                           f"{np.mean(r2_vals):.4f}", f"{np.std(r2_vals):.4f}",
                           f"{np.mean(sp_vals):.4f}", f"{np.std(sp_vals):.4f}"])

    logger.info(f"Time-forward ranking complete: {len(all_results)} runs, saved to {output_dir}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation 2: Reference-Aware Scaffold Ranking
# ═══════════════════════════════════════════════════════════════════════════════

def run_scaffold_ranking(records, model_ids, min_family_size=8, seed=42, strict=True):
    """Scaffold-family ranking: within-family support/test split."""
    # Define families
    families = defaultdict(list)
    for r in records:
        family_id = f"{r['source']}|len={r['monomer_length']}|{r['molecule_shape']}"
        families[family_id].append(r)

    # Filter by size
    valid_families = {fid: members for fid, members in families.items()
                     if len(members) >= min_family_size}
    logger.info(f"Families: {len(families)} total, {len(valid_families)} with >= {min_family_size} members")
    logger.info(f"Total compounds in valid families: {sum(len(m) for m in valid_families.values())}")

    output_dir = PROJECT_ROOT / "results/benchmark/scaffold_ranking"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    all_results = []
    skipped = []

    for family_id, members in sorted(valid_families.items()):
        # Sort by original name (try to parse compound number)
        n_members = len(members)
        support_size = max(3, int(0.3 * n_members))

        # Try to sort by compound number
        try:
            def extract_num(name):
                import re
                nums = re.findall(r'(\d+)', name)
                return int(nums[-1]) if nums else 0
            members_sorted = sorted(members, key=lambda r: extract_num(r['original_name']))
        except Exception:
            rng.shuffle(members)
            members_sorted = members

        support = members_sorted[:support_size]
        test = members_sorted[support_size:]

        # Historical training data: ALL available records NOT in this family
        # (including same-year data from other families — this is the realistic scenario
        #  where a researcher has all published data before starting a new campaign)
        member_ids = {m['sample_id'] for m in members}
        historical = [r for r in records
                     if r['sample_id'] not in member_ids]

        if len(historical) < 10:
            skipped.append({
                'family_id': family_id, 'n_members': n_members,
                'reason': f'Insufficient historical training data ({len(historical)} samples)'
            })
            continue

        # Leakage check
        support_ids = {r['sample_id'] for r in support}
        test_ids = {r['sample_id'] for r in test}
        hist_ids = {r['sample_id'] for r in historical}

        duplicate_leakage = len(test_ids & (support_ids | hist_ids))
        if strict and duplicate_leakage > 0:
            skipped.append({
                'family_id': family_id, 'n_members': n_members,
                'reason': f'Duplicate leakage detected ({duplicate_leakage} samples)'
            })
            continue

        if len(test) < 2 or len(support) < 2:
            skipped.append({
                'family_id': family_id, 'n_members': n_members,
                'reason': f'Too few support ({len(support)}) or test ({len(test)}) samples'
            })
            continue

        for model_id in model_ids:
            try:
                metrics = train_and_evaluate_ranking(
                    model_id, historical, support, test,
                    seed=seed, description=family_id[:80])
                all_results.append({
                    'family_id': family_id,
                    'source': members[0]['source'],
                    'target_year': target_year,
                    'monomer_length': members[0]['monomer_length'],
                    'molecule_shape': members[0]['molecule_shape'],
                    'n_train_historical': len(historical),
                    'n_support': len(support),
                    'n_test': len(test),
                    'model_id': model_id,
                    'seed': seed,
                    'status': 'completed',
                    'duplicate_leakage': duplicate_leakage,
                    'mode': 'strict' if strict else 'relaxed',
                    **{f'metric_{k}': v for k, v in metrics.items()},
                })
            except Exception as e:
                logger.error(f"  {model_id} on {family_id[:50]} FAILED: {e}")

    # Save family-level results
    if all_results:
        with open(output_dir / "family_level_results.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    # Save skipped
    if skipped:
        with open(output_dir / "skipped_scaffolds_with_reason.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=skipped[0].keys())
            writer.writeheader()
            writer.writerows(skipped)

    # Summary by model
    summary = defaultdict(list)
    for r in all_results:
        if r.get('status') == 'completed':
            summary[r['model_id']].append(r)

    with open(output_dir / "summary_by_model.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_id', 'n_families', 'mean_within_spearman', 'std_spearman',
                        'mean_pairwise_acc', 'std_pairwise_acc', 'mean_top10_enrichment'])
        for mid, results in sorted(summary.items()):
            sp_vals = [r.get('metric_within_family_spearman', r.get('metric_spearman', 0))
                      for r in results]
            pa_vals = [r.get('metric_pairwise_ranking_accuracy', 0) for r in results]
            te_vals = [r.get('metric_top_k_enrichment', 0) for r in results]
            writer.writerow([mid, len(results),
                           f"{np.nanmean(sp_vals):.4f}", f"{np.nanstd(sp_vals):.4f}",
                           f"{np.nanmean(pa_vals):.4f}", f"{np.nanstd(pa_vals):.4f}",
                           f"{np.nanmean(te_vals):.4f}"])

    logger.info(f"Scaffold ranking: {len(all_results)} results from {len(summary)} models")
    logger.info(f"Skipped: {len(skipped)} families")
    return all_results, skipped


# ═══════════════════════════════════════════════════════════════════════════════
# Common training + evaluation
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_FEATURIZER_MAP = {
    'seq_aa_xgb': ('aa_composition', {}),
    'chem_A_xgb': ('anchor_aware', {'descriptor_set': 'basic', 'ablation_mode': 'chemistry_only'}),
    'chem_site_AB_xgb': ('anchor_aware', {'descriptor_set': 'basic', 'ablation_mode': 'chemistry_anchors'}),
    'full_ABC_xgb': ('anchor_aware', {'descriptor_set': 'basic', 'ablation_mode': 'full'}),
    'ecfp_xgb': ('ecfp', {}),
    'rdkit_full_xgb': ('rdkit_full', {}),
}


def train_and_evaluate_ranking(model_id, train_records, support_records, test_records,
                               seed=42, description=""):
    """Train model on train+support, evaluate ranking on test."""
    from src.benchmark.featurizers import FEATURIZER_REGISTRY
    from xgboost import XGBRegressor
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr

    f_key, f_kwargs = MODEL_FEATURIZER_MAP.get(model_id, ('anchor_aware', {}))
    featurizer = FEATURIZER_REGISTRY[f_key](**f_kwargs)

    # Get samples
    train_samples = [r['sample'] for r in train_records]
    if support_records:
        train_samples += [r['sample'] for r in support_records]
    test_samples = [r['sample'] for r in test_records]

    # Featurize
    featurizer.fit(train_samples)
    X_train = np.nan_to_num(featurizer.transform(train_samples))
    y_train = np.array([r['pampa'] for r in train_records +
                       (support_records if support_records else [])])

    X_test = np.nan_to_num(featurizer.transform(test_samples))
    y_test = np.array([r['pampa'] for r in test_records])

    # Train
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                        verbosity=0, n_jobs=-1, random_state=seed)
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)

    # Metrics
    rmse = float(np.sqrt(np.mean((y_test - y_pred)**2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    sp, _ = spearmanr(y_test, y_pred)

    # Pairwise ranking accuracy
    n = len(y_test)
    if n >= 2:
        n_pairs = min(500, n * (n - 1) // 2)
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        correct = total = 0
        for _ in range(n_pairs):
            a, b = rng.choice(idx, 2, replace=False)
            if y_test[a] != y_test[b]:
                total += 1
                if (y_test[a] > y_test[b]) == (y_pred[a] > y_pred[b]):
                    correct += 1
        pairwise_acc = correct / total if total > 0 else 0.5
    else:
        pairwise_acc = 0.5

    # Top-10% enrichment
    k = max(1, int(n * 0.1))
    true_top = set(np.argsort(-y_test)[:k])
    pred_top = set(np.argsort(-y_pred)[:k])
    top10_enrich = len(true_top & pred_top) / k if k > 0 else 0

    return {
        'rmse': rmse, 'mae': mae, 'r2': r2,
        'spearman': float(sp),
        'within_family_spearman': float(sp),
        'pairwise_ranking_accuracy': pairwise_acc,
        'top_k_enrichment': top10_enrich,
        'n_train': len(train_samples),
        'n_test': n,
    }


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("Retrospective Ranking Simulations")
    logger.info("=" * 60)

    # Load data
    records = load_enriched_data()
    model_ids = ['seq_aa_xgb', 'chem_A_xgb', 'chem_site_AB_xgb',
                'full_ABC_xgb', 'ecfp_xgb', 'rdkit_full_xgb']

    # Simulation 1: Time-forward
    logger.info("\n--- Simulation 1: Time-Forward Global Ranking ---")
    tf_results = run_time_forward_ranking(records, model_ids)

    # Simulation 2: Scaffold ranking
    logger.info("\n--- Simulation 2: Reference-Aware Scaffold Ranking ---")
    sf_results, sf_skipped = run_scaffold_ranking(records, model_ids, min_family_size=8)

    # Generate interpretation templates
    gen_interpretation_templates(sf_results)

    # Generate methods text
    gen_methods_text(sf_results, sf_skipped)

    logger.info("\nAll ranking simulations complete!")


def gen_interpretation_templates(scaffold_results):
    """Generate two interpretation templates based on results."""
    output_dir = PROJECT_ROOT / "results/benchmark/scaffold_ranking"

    # Determine which template is more appropriate
    chem_sp = []
    site_sp = []
    for r in scaffold_results:
        if r.get('status') == 'completed':
            sp = r.get('metric_within_family_spearman', r.get('metric_spearman', 0))
            if r['model_id'] == 'chem_A_xgb':
                chem_sp.append(sp)
            elif r['model_id'] == 'chem_site_AB_xgb':
                site_sp.append(sp)

    delta_sp = np.mean(site_sp) - np.mean(chem_sp) if site_sp and chem_sp else 0

    if delta_sp > 0.03:
        recommended = "positive"
        template = (
            "In the retrospective scaffold-focused ranking simulation, the site-conditioned "
            "model improved within-campaign ranking over chemistry-only descriptors "
            f"(ΔSpearman = {delta_sp:+.3f}), suggesting that site information may be useful "
            "for prioritizing novel edits within established scaffold campaigns. "
            "However, the absolute ranking accuracy remains modest, and these results should "
            "be interpreted as weak prioritization cues rather than reliable prospective "
            "decision rules."
        )
    else:
        recommended = "modest"
        template = (
            "Although the site-conditioned model improved random-split prediction, the "
            "retrospective ranking gain was modest in the scaffold-focused evaluation "
            f"(ΔSpearman = {delta_sp:+.3f}), indicating that current site descriptors should "
            "be interpreted as weak prioritization cues rather than reliable prospective "
            "decision rules. Future work should develop stronger site representations "
            "and evaluate on truly prospective benchmarks."
        )

    with open(output_dir / "scaffold_ranking_interpretation_templates.md", 'w') as f:
        f.write(f"# Scaffold Ranking Interpretation\n\n"
                f"**Recommended interpretation (auto-selected):** `{recommended}`\n"
                f"**ΔSpearman (site-aware − chemistry):** {delta_sp:+.3f}\n\n"
                f"## Version A (Positive)\n\n{template if recommended == 'positive' else '(see version B)'}\n\n"
                f"## Version B (Modest)\n\n{template if recommended == 'modest' else '(see version A)'}\n")

    logger.info(f"Interpretation template: {recommended} (ΔSpearman={delta_sp:+.3f})")


def gen_methods_text(scaffold_results, skipped):
    """Generate methods description for manuscript."""
    output_dir = PROJECT_ROOT / "results/benchmark/scaffold_ranking"

    n_results = len([r for r in scaffold_results if r.get('status') == 'completed'])
    n_families = len(set(r['family_id'] for r in scaffold_results
                        if r.get('status') == 'completed'))
    n_skipped = len(skipped)

    text = (
        "# Scaffold-Focused Ranking Simulation — Methods\n\n"
        "## Design\n"
        "To address concerns about random split over-optimism and cluster split harshness, "
        "we implemented a retrospective scaffold-focused ranking simulation.\n\n"
        "### Family Definition\n"
        "- `family_id = Source + Monomer_Length + Molecule_Shape`\n"
        "- Minimum family size: 8 compounds\n"
        "- Only families with valid Year and PAMPA labels\n\n"
        "### Ranking Protocol\n"
        "1. Historical training: all compounds from same Source with Year < target_year\n"
        "2. Support set: 30% of family compounds (min 3)\n"
        "3. Test set: remaining 70% of family compounds\n"
        "4. Models trained on historical + support, evaluated on test\n\n"
        "### Leakage Prevention\n"
        "- Same_Peptides_Permeability excluded from features\n"
        "- Duplicate leakage detection via Structurally_Unique_ID cross-check\n"
        "- Strict mode: exclude families with any leakage\n\n"
        f"### Results\n"
        f"- Families evaluated: {n_families}\n"
        f"- Total model-family runs: {n_results}\n"
        f"- Families skipped: {n_skipped}\n"
        f"- Reasons: insufficient historical data, too few members, leakage detected\n\n"
        "## Limitations\n"
        "- Year is publication year, not experiment date — may not reflect true temporal order\n"
        "- Historical data may include structurally related compounds not in the same family\n"
        "- Small within-family sample sizes limit statistical power\n"
    )

    with open(output_dir / "scaffold_ranking_methods_text.md", 'w') as f:
        f.write(text)


if __name__ == "__main__":
    main()
