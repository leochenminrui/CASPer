#!/usr/bin/env python3
"""
run_all_supplement_5seed.py
===========================
Re-run ALL supplementary mechanistic control experiments with 5 seeds
and Optuna HPO, producing mean±SD results.

Covers:
  S1. Position-only control
  S2. Binary site-perturbation controls (correct / wrong / coarse)
  S3. Graded anchor-shift perturbation
  S4. B-subblock feature ablation

Usage:
    python3 scripts/run_all_supplement_5seed.py

Output:
    results/supplement_5seed/
"""

import sys, json, csv, copy, warnings, hashlib, re, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from data.loader import load_pem_dataset
from benchmark.featurizers import (
    AnchorAwareWrapper, SiteOnlyFeaturizer, ContextOnlyFeaturizer,
    PositionOnlyFeaturizer, AACompositionFeaturizer,
)
from benchmark.evaluation import compute_all_metrics
from benchmark.optuna_tuner import tune_xgboost

DATASET = "CycPeptMPDB_PAMPA"
SPLIT = "random"
SEEDS = [0, 1, 2, 3, 4]
N_TRIALS = 50
OUT_DIR = PROJECT / "results" / "supplement_5seed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Perturbation helpers ─────────────────────────────────────────────────────

def _hash_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2 ** 31)


def perturb_wrong_anchor(sample):
    s = copy.deepcopy(sample)
    seq_len = len(s.sequence)
    for edit in s.edits:
        if edit.anchor_kind == "global" or not edit.anchor_positions:
            continue
        orig = edit.anchor_positions[0]
        offsets = [1,-1,2,-2,3,-3,4,-4,5,-5]
        candidate = orig
        for off in offsets:
            cand = orig + off
            if 0 <= cand < seq_len:
                h = _hash_int(f"w_{s.sample_id}_{orig}_{off}")
                if h % 2 == 0:
                    candidate = cand
                    break
                candidate = cand
        if candidate == orig:
            candidate = seq_len - 1 if orig < seq_len // 2 else 0
        if 0 <= candidate < seq_len and candidate != orig:
            edit.anchor_positions = [candidate]
    return s


def perturb_coarse_position(sample):
    s = copy.deepcopy(sample)
    seq_len = max(len(s.sequence), 3)
    tert = seq_len // 3
    for edit in s.edits:
        if edit.anchor_kind == "global" or not edit.anchor_positions:
            continue
        orig = edit.anchor_positions[0]
        if orig < tert:
            coarse = tert // 2
        elif orig < 2 * tert:
            coarse = tert + tert // 2
        else:
            coarse = 2 * tert + tert // 2
        coarse = min(max(0, coarse), seq_len - 1)
        edit.anchor_positions = [coarse]
    return s


def perturb_shift_anchor(sample, shift):
    """Shift all anchors by a fixed offset (clamped to [0, seq_len-1])."""
    s = copy.deepcopy(sample)
    seq_len = len(s.sequence)
    for edit in s.edits:
        if edit.anchor_kind == "global" or not edit.anchor_positions:
            continue
        orig = edit.anchor_positions[0]
        new = max(0, min(seq_len - 1, orig + shift))
        edit.anchor_positions = [new]
    return s


# ── Training helpers ─────────────────────────────────────────────────────────

def train_and_evaluate(featurizer, train_samples, val_samples, test_samples,
                       test_label, seed):
    """Single seed: HPO → train → evaluate. Returns metrics dict."""
    featurizer.fit(train_samples)
    X_train = np.nan_to_num(featurizer.transform(train_samples), nan=0.0)
    X_val   = np.nan_to_num(featurizer.transform(val_samples),   nan=0.0)
    X_test  = np.nan_to_num(featurizer.transform(test_samples),  nan=0.0)

    y_train = np.array([s.label for s in train_samples])
    y_val   = np.array([s.label for s in val_samples])
    y_test  = np.array([s.label for s in test_samples])

    tune_result = tune_xgboost(
        X_train, y_train, X_val, y_val,
        n_trials=N_TRIALS, random_seed=seed,
        output_dir=OUT_DIR / f"tuning_{test_label}_seed{seed}",
    )
    model = tune_result.get("best_model")
    if model is None:
        from xgboost import XGBRegressor
        model = XGBRegressor(**tune_result["best_params"], random_state=seed,
                             tree_method="hist", verbosity=0, n_jobs=-1)
        model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    m = compute_all_metrics(y_test, y_pred)
    return {
        "seed": seed, "setting": test_label,
        "rmse": m["rmse"], "mae": m["mae"],
        "r2": m["r2"], "spearman": m["spearman"],
    }


# ── Ablation featurizers ─────────────────────────────────────────────────────

# We need: A, A+B1, A+B2, A+B3, A+B, A+C, A+B+C
# These use the AnchorAwareDescriptorFeaturizer with different ablation_modes

ABLATION_MODES = {
    "A":     "chemistry_only",
    "A+B1":  "chemistry_position",
    "A+B2":  "chemistry_residue",
    "A+B3":  "chemistry_context",
    "A+B":   "chemistry_anchors",
    "A+C":   "chemistry_attachment",
    "A+B+C": "full",
}


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Supplementary Mechanistic Controls — 5 seeds + Optuna")
    print(f"Dataset: {DATASET}, Split: {SPLIT}, Seeds: {SEEDS}")
    print("=" * 65)

    # Load data once
    data = load_pem_dataset(DATASET, SPLIT)
    train_raw = data["train"]
    val_raw   = data["val"]
    test_raw  = data["test"]
    print(f"Samples: train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}")

    # Pre-build perturbed test sets (deterministic)
    print("Building perturbed test sets...")
    perturbed = {
        "Correct site": test_raw,
        "Wrong site":   [perturb_wrong_anchor(s) for s in test_raw],
        "Coarse position": [perturb_coarse_position(s) for s in test_raw],
    }
    for shift in [-5, -3, -2, -1, 1, 2, 3, 5]:
        perturbed[f"Shift {shift:+d}"] = [perturb_shift_anchor(s, shift) for s in test_raw]
    print("  Done.")

    all_rows = []

    # ──────────────────────────────────────────────────────────────────────────
    # S1. Position-only control
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("S1. POSITION-ONLY CONTROL")
    print("=" * 65)
    for seed in tqdm(SEEDS, desc="Position-only"):
        fz = PositionOnlyFeaturizer()
        row = train_and_evaluate(fz, train_raw, val_raw, test_raw,
                                 "Position-only", seed)
        all_rows.append(row)
    # Also AA composition baseline for comparison
    for seed in tqdm(SEEDS, desc="AA composition"):
        fz = AACompositionFeaturizer(use_aa_composition=True,
                                     use_property_composition=True,
                                     use_basic_features=True)
        row = train_and_evaluate(fz, train_raw, val_raw, test_raw,
                                 "AA composition", seed)
        all_rows.append(row)

    # ──────────────────────────────────────────────────────────────────────────
    # S2. Binary site-perturbation controls (correct / wrong / coarse)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("S2. BINARY SITE-PERTURBATION CONTROLS")
    print("=" * 65)
    for setting_name in ["Correct site", "Wrong site", "Coarse position"]:
        for seed in tqdm(SEEDS, desc=setting_name):
            fz = AnchorAwareWrapper(descriptor_set="basic", ablation_mode="full")
            row = train_and_evaluate(fz, train_raw, val_raw,
                                     perturbed[setting_name], setting_name, seed)
            all_rows.append(row)

    # ──────────────────────────────────────────────────────────────────────────
    # S3. Graded anchor-shift perturbation
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("S3. GRADED ANCHOR-SHIFT PERTURBATION")
    print("=" * 65)
    for shift in [-5, -3, -2, -1, 1, 2, 3, 5]:
        label = f"Shift {shift:+d}"
        for seed in tqdm(SEEDS, desc=label):
            fz = AnchorAwareWrapper(descriptor_set="basic", ablation_mode="full")
            row = train_and_evaluate(fz, train_raw, val_raw,
                                     perturbed[label], label, seed)
            all_rows.append(row)

    # ──────────────────────────────────────────────────────────────────────────
    # S4. B-subblock feature ablation
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("S4. B-SUBBLOCK FEATURE ABLATION")
    print("=" * 65)
    for fs_label, ablation_mode in ABLATION_MODES.items():
        for seed in tqdm(SEEDS, desc=fs_label):
            fz = AnchorAwareWrapper(descriptor_set="basic",
                                    ablation_mode=ablation_mode)
            row = train_and_evaluate(fz, train_raw, val_raw, test_raw,
                                     fs_label, seed)
            all_rows.append(row)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "all_per_seed.csv", index=False)
    print(f"\nPer-seed results: {OUT_DIR / 'all_per_seed.csv'}")

    summary = df.groupby("setting").agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"),
    ).reset_index()

    summary.to_csv(OUT_DIR / "all_summary.csv", index=False)

    # ── Print formatted tables ────────────────────────────────────────────────
    for section_name, settings in [
        ("S1. Position-only control", ["Position-only", "AA composition", "Correct site"]),
        ("S2. Binary site-perturbation", ["Correct site", "Wrong site", "Coarse position"]),
        ("S3. Graded anchor-shift", ["Correct site"] +
         [f"Shift {s:+d}" for s in [-1, 1, -2, 2, -3, 3, -5, 5]]),
        ("S4. B-subblock ablation", list(ABLATION_MODES.keys())),
    ]:
        print(f"\n{'='*65}")
        print(f"{section_name}")
        print(f"{'='*65}")
        sub = summary[summary["setting"].isin(settings)].copy()
        if len(sub) == 0:
            print("  (no results)")
            continue
        print(f"{'Setting':25s}  {'RMSE':>18s}  {'MAE':>18s}  {'R²':>18s}  {'Spearman':>18s}")
        print("-" * 100)
        for s in settings:
            r = sub[sub["setting"] == s]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            print(f"{s:25s}  {r['rmse_mean']:.4f}±{r['rmse_std']:.4f}  "
                  f"{r['mae_mean']:.4f}±{r['mae_std']:.4f}  "
                  f"{r['r2_mean']:.4f}±{r['r2_std']:.4f}  "
                  f"{r['spearman_mean']:.4f}±{r['spearman_std']:.4f}")

    print(f"\nDone. Full results: {OUT_DIR / 'all_summary.csv'}")


if __name__ == "__main__":
    main()
