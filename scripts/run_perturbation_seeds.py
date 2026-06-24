#!/usr/bin/env python3
"""
run_perturbation_seeds.py
=========================
Run site-perturbation control experiments with 5 seeds to get mean±SD.

Perturbations (edit chemistry held fixed, only site annotations degraded):
  1. Correct site  — unmodified anchor positions
  2. Wrong site    — each anchor shifted to a different valid position
  3. Coarse position — each anchor mapped to its tertile region center

Usage:
    python3 scripts/run_perturbation_seeds.py

Output:
    results/perturbation/perturbation_5seed_summary.csv
    results/perturbation/per_seed_results.csv
"""

import sys, json, csv, warnings, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from data.loader import load_pem_dataset
from benchmark.featurizers import AnchorAwareWrapper
from benchmark.evaluation import compute_all_metrics
from benchmark.optuna_tuner import tune_xgboost

# ── Config ───────────────────────────────────────────────────────────────────
DATASET = "CycPeptMPDB_PAMPA"
SPLIT = "random"
SEEDS = [0, 1, 2, 3, 4]
N_TRIALS = 50
OUT_DIR = PROJECT / "results" / "perturbation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Anchor perturbation helpers ──────────────────────────────────────────────

def _deterministic_hash_int(seed_str: str) -> int:
    """Return a deterministic 32-bit integer from a string."""
    h = hashlib.md5(seed_str.encode()).hexdigest()
    return int(h, 16) % (2 ** 31)


def perturb_wrong_anchor(sample):
    """
    Return a perturbed copy of the sample with wrong anchor positions.

    For each edit, pick a deterministic different valid position.
    """
    import copy
    s = copy.deepcopy(sample)
    seq_len = len(s.sequence)
    for edit in s.edits:
        if edit.anchor_kind == "global" or not edit.anchor_positions:
            continue
        orig = edit.anchor_positions[0]
        h = _deterministic_hash_int(f"wrong_{s.sample_id}_{orig}")
        offsets = [1,-1,2,-2,3,-3,4,-4,5,-5]
        candidate = orig
        for off in offsets:
            cand = orig + off
            if 0 <= cand < seq_len:
                h += 1
                if (h % (len(offsets) + 1)) != 0:
                    candidate = cand
                    break
                candidate = cand  # use last valid as fallback
        if candidate == orig:
            candidate = seq_len - 1 if orig < seq_len // 2 else 0
        if 0 <= candidate < seq_len and candidate != orig:
            edit.anchor_positions = [candidate]
    return s


def perturb_coarse_position(sample):
    """
    Return a perturbed copy with anchor positions mapped to tertile centres.
    """
    import copy
    s = copy.deepcopy(sample)
    seq_len = max(len(s.sequence), 3)
    tertile_size = seq_len // 3
    for edit in s.edits:
        if edit.anchor_kind == "global" or not edit.anchor_positions:
            continue
        orig = edit.anchor_positions[0]
        if orig < tertile_size:
            coarse = tertile_size // 2
        elif orig < 2 * tertile_size:
            coarse = tertile_size + tertile_size // 2
        else:
            coarse = 2 * tertile_size + tertile_size // 2
        coarse = min(max(0, coarse), seq_len - 1)
        edit.anchor_positions = [coarse]
    return s


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Site Perturbation Controls — 5 seeds")
    print(f"Dataset: {DATASET}, Split: {SPLIT}")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    # Load data once
    data = load_pem_dataset(DATASET, SPLIT)
    train_samples = data["train"]
    val_samples = data["val"]
    test_samples = data["test"]
    print(f"Samples: train={len(train_samples)}, val={len(val_samples)}, "
          f"test={len(test_samples)}")

    # Prepare perturbed test sets (deterministic, same across seeds)
    print("Preparing perturbed test sets...")
    test_wrong  = [perturb_wrong_anchor(s) for s in test_samples]
    test_coarse = [perturb_coarse_position(s) for s in test_samples]
    print("  Done.")

    all_rows = []

    for seed in SEEDS:
        print(f"\n── Seed {seed} ──")

        # Featurizer: full A+B+C model
        featurizer = AnchorAwareWrapper(
            descriptor_set="basic", ablation_mode="full")
        featurizer.fit(train_samples)

        X_train = featurizer.transform(train_samples)
        X_val   = featurizer.transform(val_samples)
        X_test  = featurizer.transform(test_samples)
        X_wrong = featurizer.transform(test_wrong)
        X_coarse = featurizer.transform(test_coarse)

        y_train = np.array([s.label for s in train_samples])
        y_val   = np.array([s.label for s in val_samples])
        y_test  = np.array([s.label for s in test_samples])

        # NaN handling
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val   = np.nan_to_num(X_val,   nan=0.0)
        X_test  = np.nan_to_num(X_test,  nan=0.0)
        X_wrong  = np.nan_to_num(X_wrong,  nan=0.0)
        X_coarse = np.nan_to_num(X_coarse, nan=0.0)

        # HPO
        tune_result = tune_xgboost(
            X_train, y_train, X_val, y_val,
            n_trials=N_TRIALS,
            random_seed=seed,
            output_dir=OUT_DIR / f"seed_{seed}",
        )
        best_params = tune_result["best_params"]
        val_metrics = tune_result["best_val_metrics"]
        model = tune_result.get("best_model")

        if model is None:
            from xgboost import XGBRegressor
            model = XGBRegressor(**best_params, random_state=seed,
                                 tree_method="hist", verbosity=0, n_jobs=-1)
            model.fit(X_train, y_train, verbose=False)

        def evaluate(X, y, setting_name):
            y_pred = model.predict(X)
            m = compute_all_metrics(y, y_pred)
            return {
                "seed": seed, "setting": setting_name,
                "rmse": m["rmse"], "mae": m["mae"],
                "r2": m["r2"], "spearman": m["spearman"],
                "pearson": m.get("pearson", np.nan),
            }

        # Evaluate all three settings
        for setting_name, X_pert in [("Correct site", X_test),
                                       ("Wrong site", X_wrong),
                                       ("Coarse position", X_coarse)]:
            row = evaluate(X_pert, y_test, setting_name)
            all_rows.append(row)
            print(f"  {setting_name:20s}  R²={row['r2']:.4f}  "
                  f"RMSE={row['rmse']:.4f}  Spearman={row['spearman']:.4f}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    per_seed_path = OUT_DIR / "perturbation_per_seed.csv"
    df.to_csv(per_seed_path, index=False)
    print(f"\nPer-seed results saved: {per_seed_path}")

    summary = df.groupby("setting").agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"),
    ).reset_index()

    summary_path = OUT_DIR / "perturbation_5seed_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("SUMMARY (mean ± SD across 5 seeds)")
    print("=" * 60)
    for _, r in summary.iterrows():
        print(f"\n{r['setting']}:")
        print(f"  RMSE     = {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}")
        print(f"  MAE      = {r['mae_mean']:.4f} ± {r['mae_std']:.4f}")
        print(f"  R²       = {r['r2_mean']:.4f} ± {r['r2_std']:.4f}")
        print(f"  Spearman = {r['spearman_mean']:.4f} ± {r['spearman_std']:.4f}")

    print(f"\nFull results: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
