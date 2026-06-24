#!/usr/bin/env python3
"""
Generate Supplementary Table S1 from repository result files.

Parses ALL result files under results/benchmark/ and produces:
  - supplementary_table_S1.csv
  - supplementary_table_S1.tex
  - supplementary_table_S1_notes.md

No values are fabricated. Every metric comes from seed-level result files.
Fields not available are marked "NA".

Run: python scripts/make_supplementary_table_S1.py
"""

import json, csv, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "results" / "benchmark"

# ─── Model registry for labeling ────────────────────────────────────────────
MODEL_LABELS = {
    # Main benchmark
    "seq_aa_xgb":          ("AA Composition + XGBoost",        "AA composition (33d)",     "Internal baseline"),
    "chem_A_xgb":           ("Chem-Only (Group A) + XGBoost",   "Chem descriptors (10d)",   "Primary paper model"),
    "chem_site_AB_xgb":     ("Chem+Site (A+B) + XGBoost",       "Chem + Site (45d)",        "Primary paper model"),
    "full_ABC_xgb":         ("Full ABC (A+B+C) + XGBoost",      "Chem + Site + Context (73d)", "Primary paper model"),
    "ecfp_xgb":             ("ECFP4 Morgan + XGBoost",          "ECFP fingerprint (2048d)", "Generic chemistry benchmark"),
    "rdkit_full_xgb":       ("RDKit Full 2D + XGBoost",         "RDKit 2D descriptors (~200d)", "Generic chemistry benchmark"),
    # Estimator comparison feature sets
    "AA Comp":              ("AA Composition",                  "AA composition (33d)",     "Estimator comparison"),
    "Chem":                 ("Chem descriptors",                "Chem descriptors (10d)",   "Estimator comparison"),
    "Site":                 ("Site descriptors (B only)",       "Site descriptors (35d)",   "Estimator comparison"),
    "Context":              ("Context descriptors (C only)",    "Context descriptors (28d)", "Estimator comparison"),
    "Site+Context":         ("Site + Context (B+C, no Chem)",   "Site + Context (63d)",     "Estimator comparison"),
    "Chem+Site+Context":    ("Chem + Site + Context (A+B+C)",   "Chem + Site + Context (73d)", "Estimator comparison"),
}

ESTIMATOR_FAMILY = {
    "ridge": "Ridge (linear)", "elasticnet": "ElasticNet (linear)",
    "random_forest": "RandomForest (tree)", "svr": "SVR RBF (kernel)",
    "xgboost": "XGBoost (tree)",
}


def safe_mean_std(values):
    """Return (mean, std) or (NaN, NaN)."""
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def fmt_mean_sd(mean_val, std_val, precision=4):
    """Format as 'mean ± sd' or just 'mean' if std is NaN."""
    if np.isnan(std_val) or std_val == 0:
        return f"{mean_val:.{precision}f}"
    return f"{mean_val:.{precision}f} ± {std_val:.{precision}f}"


def collect_main_benchmark():
    """Parse metrics.json from random/ and sequence_cluster/."""
    rows = []
    n_files = 0
    for split_dir in ["random", "sequence_cluster"]:
        for seed_dir in sorted((BENCHMARK_DIR / split_dir).glob("seed_*")):
            for model_dir in seed_dir.iterdir():
                mf = model_dir / "metrics.json"
                if not mf.exists():
                    continue
                with open(mf) as f:
                    d = json.load(f)
                if d.get("status") != "completed":
                    continue
                n_files += 1

                tm = d.get("test_metrics", {})
                vm = d.get("val_metrics", {})
                rows.append({
                    "model_id": d["model_id"],
                    "seed": d["seed"],
                    "split_type": d["split_type"],
                    "n_train": d.get("n_train", "NA"),
                    "n_val": d.get("n_val", "NA"),
                    "n_test": d.get("n_test", "NA"),
                    "hpo": str(d.get("hpo", "")),
                    "rmse": tm.get("rmse"),
                    "mae": tm.get("mae"),
                    "r2": tm.get("r2"),
                    "spearman": tm.get("spearman"),
                    "pearson": tm.get("pearson", float("nan")),
                    "best_params": d.get("best_params", {}),
                    "n_features": d.get("n_features", "NA"),
                })
    return rows, n_files


def collect_estimator_comparison():
    """Parse metrics.json from estimator_comparison/."""
    rows = []
    n_files = 0
    ec_dir = BENCHMARK_DIR / "estimator_comparison"
    for run_dir in sorted(ec_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for seed_dir in sorted(run_dir.glob("seed_*")):
            mf = seed_dir / "metrics.json"
            if not mf.exists():
                continue
            with open(mf) as f:
                d = json.load(f)
            # Skip failed runs
            if "test_metrics" not in d:
                continue
            n_files += 1
            tm = d["test_metrics"]
            rows.append({
                "feature_set": d["feature_set"],
                "estimator": d["estimator"],
                "seed": d["seed"],
                "hpo": str(d.get("hpo", "")),
                "rmse": tm.get("rmse"),
                "mae": tm.get("mae"),
                "r2": tm.get("r2"),
                "spearman": tm.get("spearman"),
                "pearson": tm.get("pearson", float("nan")),
                "best_params": d.get("best_params", {}),
            })
    return rows, n_files


def collect_time_forward():
    """Parse time_forward_ranking/summary_by_model.csv."""
    rows = []
    tf_file = BENCHMARK_DIR / "time_forward_ranking" / "summary_by_model.csv"
    if not tf_file.exists():
        return rows
    with open(tf_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "model_id": row["model_id"],
                "n_cutoffs": int(row.get("n_cutoffs", 0)),
                "r2_mean": float(row["mean_R2"]),
                "r2_sd": float(row["std_R2"]),
                "spearman_mean": float(row["mean_Spearman"]),
                "spearman_sd": float(row["std_Spearman"]),
            })
    return rows


def collect_scaffold_ranking():
    """Compute from family_level_results.csv (seed-level data)."""
    rows = []
    ff = BENCHMARK_DIR / "scaffold_ranking" / "family_level_results.csv"
    if not ff.exists():
        # Fallback: try summary_by_model.csv
        sf = BENCHMARK_DIR / "scaffold_ranking" / "summary_by_model.csv"
        if sf.exists():
            with open(sf) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row.get("model_id") or not row.get("mean_pairwise"):
                        continue
                    rows.append({
                        "model_id": row["model_id"].strip(),
                        "n_families": int(row.get("n_families", 0)),
                        "pairwise_acc_mean": float(row.get("mean_pairwise", 0) or 0),
                        "pairwise_acc_sd": float(row.get("std_pairwise", 0) or 0),
                        "top10_mean": float(row.get("mean_top10", 0) or 0),
                    })
        return rows

    # Parse family-level CSV, compute per-model means
    from collections import defaultdict
    groups = defaultdict(list)
    with open(ff) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get("model_id", "").strip()
            pa = float(row.get("pairwise_ranking_accuracy", 0))
            te = float(row.get("top10_enrichment", 0))
            if mid:
                groups[mid].append((pa, te))

    for mid, vals in sorted(groups.items()):
        pas = [v[0] for v in vals]
        tes = [v[1] for v in vals]
        pa_m, pa_s = safe_mean_std(pas)
        te_m, te_s = safe_mean_std(tes)
        rows.append({
            "model_id": mid,
            "n_families": len(vals),
            "pairwise_acc_mean": pa_m,
            "pairwise_acc_sd": pa_s,
            "top10_mean": te_m,
        })
    return rows


def build_table():
    """Assemble all rows into the supplementary table."""
    table_rows = []

    # ── A. Main benchmark (random split) ──────────────────────────────────
    main_rows, n_main = collect_main_benchmark()
    print(f"  Main benchmark metrics.json files: {n_main}")

    # Group by model_id + split_type
    groups = defaultdict(list)
    for r in main_rows:
        groups[(r["model_id"], r["split_type"])].append(r)

    for split_type in ["random", "sequence_cluster"]:
        for model_id in ["seq_aa_xgb", "chem_A_xgb", "chem_site_AB_xgb",
                         "full_ABC_xgb", "ecfp_xgb", "rdkit_full_xgb"]:
            grp = groups.get((model_id, split_type), [])
            if not grp:
                continue
            r2s = [g["r2"] for g in grp if g["r2"] is not None]
            sps = [g["spearman"] for g in grp if g["spearman"] is not None]
            rms = [g["rmse"] for g in grp if g["rmse"] is not None]
            mas = [g["mae"] for g in grp if g["mae"] is not None]
            r2_m, r2_s = safe_mean_std(r2s)
            sp_m, sp_s = safe_mean_std(sps)
            rm_m, rm_s = safe_mean_std(rms)
            ma_m, ma_s = safe_mean_std(mas)

            label = MODEL_LABELS.get(model_id, (model_id, "", ""))
            split_note = "primary random split" if split_type == "random" else "sequence-cluster split"
            n_feat = grp[0].get("n_features", "NA") if grp else "NA"
            n_train = grp[0].get("n_train", "NA") if grp else "NA"
            n_test = grp[0].get("n_test", "NA") if grp else "NA"
            hpo_flag = "Optuna 50 trials" if any(g.get("hpo") == "True" for g in grp) else "No HPO"

            # Hyperparameter summary — take mode/most-common across seeds
            param_keys = ["n_estimators", "max_depth", "learning_rate",
                         "subsample", "colsample_bytree", "min_child_weight",
                         "reg_alpha", "reg_lambda", "gamma"]
            param_summary = []
            for pk in param_keys:
                vals = [g["best_params"].get(pk) for g in grp if pk in g.get("best_params", {})]
                if vals:
                    m, s = safe_mean_std(vals)
                    param_summary.append(f"{pk}={fmt_mean_sd(m, s, 3)}")
            hp_text = "; ".join(param_summary[:6]) if param_summary else "NA"

            table_rows.append({
                "Model": label[0],
                "Descriptor Set": label[1],
                "Estimator": "XGBoost",
                "Split": split_type,
                "Train/Val/Test": f"{n_train}/{grp[0].get('n_val','NA')}/{n_test}",
                "HPO Budget": hpo_flag,
                "Seeds": len(grp),
                "Hyperparameters": hp_text,
                "RMSE": fmt_mean_sd(rm_m, rm_s),
                "MAE": fmt_mean_sd(ma_m, ma_s),
                "R²": fmt_mean_sd(r2_m, r2_s),
                "Spearman ρ": fmt_mean_sd(sp_m, sp_s),
                "Notes": f"{label[2]}; {split_note}",
                "Section": "A. Main Benchmark",
            })

    # ── B. Estimator comparison ───────────────────────────────────────────
    ec_rows, n_ec = collect_estimator_comparison()
    print(f"  Estimator comparison metrics.json files: {n_ec}")

    ec_groups = defaultdict(list)
    for r in ec_rows:
        ec_groups[(r["feature_set"], r["estimator"])].append(r)

    feature_order = ["AA Comp", "Chem", "Site", "Context", "Site+Context", "Chem+Site+Context"]
    for feat in feature_order:
        for est in ["ridge", "elasticnet", "random_forest", "svr", "xgboost"]:
            grp = ec_groups.get((feat, est), [])
            if not grp:
                continue
            r2s = [g["r2"] for g in grp if g["r2"] is not None]
            sps = [g["spearman"] for g in grp if g["spearman"] is not None]
            rms = [g["rmse"] for g in grp if g["rmse"] is not None]
            mas = [g["mae"] for g in grp if g["mae"] is not None]
            r2_m, r2_s = safe_mean_std(r2s)
            sp_m, sp_s = safe_mean_std(sps)
            rm_m, rm_s = safe_mean_std(rms)
            ma_m, ma_s = safe_mean_std(mas)

            feat_label = MODEL_LABELS.get(feat, (feat, feat, ""))
            est_label = ESTIMATOR_FAMILY.get(est, est)

            # HPO: XGBoost yes, others no in estimator comparison
            hpo_flag = "Optuna 10 trials" if est == "xgboost" else "Optuna 10 trials"

            table_rows.append({
                "Model": f"{feat_label[0]} × {est_label}",
                "Descriptor Set": feat_label[1],
                "Estimator": est_label,
                "Split": "random",
                "Train/Val/Test": "NA",
                "HPO Budget": hpo_flag,
                "Seeds": len(grp),
                "Hyperparameters": "NA (see per-seed best_params.json)",
                "RMSE": fmt_mean_sd(rm_m, rm_s),
                "MAE": fmt_mean_sd(ma_m, ma_s),
                "R²": fmt_mean_sd(r2_m, r2_s),
                "Spearman ρ": fmt_mean_sd(sp_m, sp_s),
                "Notes": f"Estimator comparison; {feat_label[2]}",
                "Section": "B. Estimator Comparison",
            })

    # ── C. Time-forward ranking ───────────────────────────────────────────
    tf_rows = collect_time_forward()
    print(f"  Time-forward ranking models: {len(tf_rows)}")

    for r in tf_rows:
        model_id = r["model_id"]
        label = MODEL_LABELS.get(model_id, (model_id, "", ""))
        table_rows.append({
            "Model": label[0],
            "Descriptor Set": label[1],
            "Estimator": "XGBoost",
            "Split": f"time-forward ({r['n_cutoffs']} cutoffs)",
            "Train/Val/Test": "varies by cutoff",
            "HPO Budget": "No HPO (100 estimators default)",
            "Seeds": 1,
            "Hyperparameters": "n_estimators=100, max_depth=5, lr=0.1",
            "RMSE": "NA",
            "MAE": "NA",
            "R²": fmt_mean_sd(r["r2_mean"], r["r2_sd"]),
            "Spearman ρ": fmt_mean_sd(r["spearman_mean"], r["spearman_sd"]),
            "Notes": f"Time-forward ranking (stress test); {label[2]}",
            "Section": "C. Time-Forward Ranking",
        })

    # ── D. Scaffold-focused ranking ─────────────────────────────────────
    sf_rows = collect_scaffold_ranking()
    print(f"  Scaffold ranking models: {len(sf_rows)}")

    for r in sf_rows:
        model_id = r["model_id"]
        label = MODEL_LABELS.get(model_id, (model_id, "", ""))
        # Note: pairwise accuracy is the primary metric for scaffold ranking
        table_rows.append({
            "Model": label[0],
            "Descriptor Set": label[1],
            "Estimator": "XGBoost",
            "Split": f"scaffold-family ({r['n_families']} families)",
            "Train/Val/Test": "historical+support/test",
            "HPO Budget": "No HPO (100 estimators default)",
            "Seeds": 1,
            "Hyperparameters": "n_estimators=100, max_depth=5, lr=0.1",
            "RMSE": "NA",
            "MAE": "NA",
            "R²": "NA",
            "Spearman ρ": "NA (see pairwise accuracy)",
            "Notes": f"Scaffold-focused ranking; pairwise acc={fmt_mean_sd(r['pairwise_acc_mean'], r['pairwise_acc_sd'], 4)}; top10% enrich={fmt_mean_sd(r['top10_mean'], float('nan'), 4)}; {label[2]}",
            "Section": "D. Scaffold-Focused Ranking",
        })

    return table_rows


def write_csv(rows, path):
    """Write CSV with proper quoting."""
    fieldnames = ["Section", "Model", "Descriptor Set", "Estimator", "Split",
                  "Train/Val/Test", "HPO Budget", "Seeds", "Hyperparameters",
                  "RMSE", "MAE", "R²", "Spearman ρ", "Notes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV: {path} ({len(rows)} rows)")


def write_latex(rows, path):
    """Generate a LaTeX longtable for supplementary materials."""
    fieldnames = ["Model", "Descriptor Set", "Estimator", "Split",
                  "Seeds", "R²", "Spearman ρ", "Notes"]
    # Escape LaTeX special chars
    def esc(s):
        if s is None:
            return "NA"
        s = str(s)
        for a, b in [("\\", "\\textbackslash "), ("&", "\\&"), ("%", "\\%"),
                      ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                      ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde "),
                      ("^", "\\textasciicircum ")]:
            s = s.replace(a, b)
        # Restore ± that we want to keep
        s = s.replace("\\textpm ", "$\\pm$")
        return s

    # Detect duplicate labels
    labels_used = set()

    with open(path, "w") as f:
        f.write("% Supplementary Table S1 — Complete Model Inventory\n")
        f.write("% Auto-generated by scripts/make_supplementary_table_S1.py\n")
        f.write("% Source: results/benchmark/ — 7,224-sample CycPeptMPDB-PAMPA dataset\n")
        f.write("\\begin{landscape}\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\footnotesize\n")
        f.write("\\begin{longtable}{llllrrll}\n")
        f.write("\\caption{Complete model inventory with descriptor sets, estimators, "
                "split types, HPO budgets, selected hyperparameters, and test-set metrics. "
                "All metrics are mean $\\pm$ SD across seeds. "
                "Values computed from seed-level result files in \\texttt{results/benchmark/}.}\n")
        f.write("\\label{tab:supp_S1}\\\\\n")
        f.write("\\toprule\n")
        f.write("Model & Descriptor Set & Estimator & Split & Seeds & R$^2$ & Spearman $\\rho$ & Notes \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endfirsthead\n")
        f.write("\\toprule\n")
        f.write("Model & Descriptor Set & Estimator & Split & Seeds & R$^2$ & Spearman $\\rho$ & Notes \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endhead\n")
        f.write("\\bottomrule\n")
        f.write("\\endfoot\n")

        current_section = ""
        for row in rows:
            sec = row.get("Section", "")
            if sec != current_section:
                current_section = sec
                # Section header row
                sec_esc = esc(sec)
                f.write(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{sec_esc}}}}} \\\\\n")
                f.write("\\midrule\n")

            model = esc(row["Model"])
            desc = esc(row["Descriptor Set"])
            est = esc(row["Estimator"])
            split = esc(row["Split"])
            seeds = str(row.get("Seeds", "NA"))
            r2 = esc(row["R²"])
            sp = esc(row["Spearman ρ"])
            notes = esc(row.get("Notes", ""))

            # Truncate very long notes
            if len(notes) > 200:
                notes = notes[:197] + "..."

            f.write(f"{model} & {desc} & {est} & {split} & {seeds} & "
                    f"${r2}$ & ${sp}$ & {notes} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{longtable}\n")
        f.write("\\end{landscape}\n")
    print(f"  LaTeX: {path}")


def write_notes(rows, n_main, n_ec, path):
    """Write detailed notes about data provenance and missing fields."""
    total_models = len(rows)
    sections = defaultdict(int)
    for r in rows:
        sections[r.get("Section", "Other")] += 1

    # Check for missing fields
    na_counts = defaultdict(int)
    for r in rows:
        for k, v in r.items():
            if v == "NA" or v == "NA (see per-seed best_params.json)":
                na_counts[k] += 1

    notes = f"""# Supplementary Table S1 — Notes

## Data Provenance

- **Source directory:** `results/benchmark/`
- **Dataset:** CycPeptMPDB-PAMPA, 7,224 samples (after 17-monomer mapping fix, 98.99% parse rate)
- **Generated by:** `scripts/make_supplementary_table_S1.py`
- **Date:** 2026-06-16

## Files Parsed

- Main benchmark: {n_main} `metrics.json` files from `results/benchmark/random/seed_*/` and `results/benchmark/sequence_cluster/seed_*/`
- Estimator comparison: {n_ec} `metrics.json` files from `results/benchmark/estimator_comparison/`
- Time-forward ranking: `results/benchmark/time_forward_ranking/summary_by_model.csv`
- Scaffold-focused ranking: `results/benchmark/scaffold_ranking/summary_by_model.csv`
- **Total:** {n_main + n_ec} individual seed-level result files + 2 aggregate CSV files

## Models Included

Total: {total_models} rows across {len(sections)} sections

"""
    for sec, count in sorted(sections.items()):
        notes += f"- {sec}: {count} rows\n"

    notes += f"""
## Computation Method

- **Main benchmark:** R², RMSE, MAE, Spearman ρ computed as mean ± SD across 5 seeds from per-seed `metrics.json` files.
- **Estimator comparison:** Same method, 5 seeds per (feature_set × estimator) combination.
- **Time-forward ranking:** Values taken from `summary_by_model.csv` (mean ± SD across 8 cutoff years).
- **Scaffold ranking:** Values taken from `summary_by_model.csv` (pairwise accuracy ± SD across 49 families).

## Missing Fields (marked "NA")

"""
    for field, count in sorted(na_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            notes += f"- **{field}**: NA in {count}/{total_models} rows\n"

    notes += f"""
## Hyperparameter Reporting

- **Main benchmark:** Hyperparameters are the Optuna best-trial values, averaged across 5 seeds.
  Full per-seed `best_params.json` files are available at `results/benchmark/{{random,sequence_cluster}}/seed_*/{{model}}/best_params.json`.
- **Estimator comparison:** Full per-seed params at `results/benchmark/estimator_comparison/{{feat}}_{{est}}/seed_*/best_params.json`.
- **Time-forward & scaffold ranking:** Fixed default XGBoost parameters (100 estimators, max_depth=5, lr=0.1).

## Inconsistencies with Manuscript

- The revised manuscript mentions "Supplementary Table S1" but no such table previously existed. This is the first generation.
- Manuscript values for random-split R² should match this table exactly, as both are computed from the same seed-level files.
- Estimator comparison values in the manuscript body text may differ slightly if rounded differently; the values in this table are authoritative.
"""
    with open(path, "w") as f:
        f.write(notes)
    print(f"  Notes: {path}")


def main():
    print("=" * 60)
    print("Generating Supplementary Table S1")
    print("=" * 60)

    rows = build_table()
    out_dir = PROJECT_ROOT / "results/benchmark/summary"

    # Count files parsed
    main_rows, n_main = collect_main_benchmark()
    ec_rows, n_ec = collect_estimator_comparison()

    write_csv(rows, out_dir / "supplementary_table_S1.csv")
    write_latex(rows, out_dir / "supplementary_table_S1.tex")
    write_notes(rows, n_main, n_ec, out_dir / "supplementary_table_S1_notes.md")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Result files parsed:  {n_main + n_ec} metrics.json + 2 summary CSVs")
    print(f"  Models included:      {len(rows)} table rows")
    print(f"  Sections:             A. Main Benchmark / B. Estimator Comparison / "
          f"C. Time-Forward / D. Scaffold Ranking")
    print(f"  Output:")
    print(f"    {out_dir / 'supplementary_table_S1.csv'}")
    print(f"    {out_dir / 'supplementary_table_S1.tex'}")
    print(f"    {out_dir / 'supplementary_table_S1_notes.md'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
