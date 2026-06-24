#!/usr/bin/env python3
"""
Fast Estimator Comparison — scikit-learn defaults for Ridge/ElasticNet/RF/SVR,
Optuna only for XGBoost. 3 seeds, 5 trials for XGBoost HPO.

Purpose: show that anchor-aware > chemistry-only > sequence-only trend
holds across model families, without the cost of full Optuna per estimator.

Runtime: ~5 min total.
"""

import sys, json, csv, logging
from pathlib import Path
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data.loader import load_pem_dataset
from benchmark.featurizers import FEATURIZER_REGISTRY

SEEDS = [0, 1, 2, 3, 4]  # same as main benchmark
N_TRIALS = 10             # same HPO budget for every estimator
OUTPUT_DIR = PROJECT_ROOT / 'results/benchmark/estimator_comparison'


# ─── Estimator definitions ──────────────────────────────────────────────────
# (label, family, model_cls, default_params, needs_scale)
ESTIMATORS = {
    'ridge':       ('Ridge',       'linear', Ridge(alpha=1.0),               {}, True),
    'elasticnet':  ('ElasticNet',  'linear', ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000), {}, True),
    'random_forest':('RandomForest','tree',  RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1), {}, False),
    'svr':         ('SVR(RBF)',    'kernel', SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1, max_iter=5000), {}, True),
    'xgboost':     ('XGBoost',     'tree',   None, {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.1,
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'tree_method': 'hist', 'verbosity': 0, 'n_jobs': -1}, False),
}

# Feature sets for comparison
# Naming: Chem=Group A, Site=Group B, Context=Group C
FEATURES = {
    'AA Comp':          ('aa_composition', {'use_aa_composition': True, 'use_property_composition': True,
                                            'use_basic_features': True, 'use_dipeptide': False}),
    'Chem':             ('anchor_aware',   {'descriptor_set': 'basic', 'ablation_mode': 'chemistry_only'}),
    'Site':             ('site_only',      {}),
    'Context':          ('context_only',   {}),
    'Site+Context':     ('anchor_aware',   {'descriptor_set': 'basic', 'ablation_mode': 'site_context_only'}),
    'Chem+Site+Context':('anchor_aware',   {'descriptor_set': 'basic', 'ablation_mode': 'full'}),
}


def compute_metrics(y_true, y_pred):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'spearman': float(spearmanr(y_true, y_pred)[0]),
    }


def run_one(est_key, feat_key, seed):
    """Train & evaluate. ALL estimators use Optuna HPO with equal budget."""
    label, family, model_cls, default_params, needs_scale = ESTIMATORS[est_key]
    feat_key_full, feat_kwargs = FEATURES[feat_key]

    # Safe filename: replace spaces/plus signs
    safe_key = feat_key.replace(' ','_').replace('+','_')
    model_dir = OUTPUT_DIR / f"{safe_key}_{est_key}" / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load
    data = load_pem_dataset("CycPeptMPDB_PAMPA", "random")
    fzr = FEATURIZER_REGISTRY[feat_key_full](**feat_kwargs)
    X_train = np.nan_to_num(fzr.transform(data['train']))
    X_val   = np.nan_to_num(fzr.transform(data['val']))
    X_test  = np.nan_to_num(fzr.transform(data['test']))
    y_train = np.array([s.label for s in data['train']])
    y_val   = np.array([s.label for s in data['val']])
    y_test  = np.array([s.label for s in data['test']])

    # Scale if needed
    if needs_scale:
        scaler = StandardScaler().fit(X_train)
        X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    import optuna, warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Common objective wrapper ─────────────────────────────────────────
    def _tune(objective_fn, study_name):
        study = optuna.create_study(direction='minimize', study_name=study_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        study.optimize(objective_fn, n_trials=N_TRIALS, show_progress_bar=False)
        return study.best_params

    # ── Per-estimator search spaces ───────────────────────────────────────
    if est_key == 'xgboost':
        def obj(trial):
            m = XGBRegressor(
                n_estimators=trial.suggest_int('n_estimators', 100, 1000),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
                reg_alpha=trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
                tree_method='hist', verbosity=0, n_jobs=-1, random_state=seed)
            m.fit(X_train, y_train, verbose=False)
            return float(np.sqrt(np.mean((y_val - m.predict(X_val))**2)))
        best_params = _tune(obj, f"xgb_{feat_key}_s{seed}")
        model = XGBRegressor(tree_method='hist', verbosity=0, n_jobs=-1, random_state=seed, **best_params)

    elif est_key == 'ridge':
        def obj(trial):
            m = Ridge(alpha=trial.suggest_float('alpha', 0.01, 1e4, log=True), random_state=seed)
            m.fit(X_train, y_train)
            return float(np.sqrt(np.mean((y_val - m.predict(X_val))**2)))
        best_params = _tune(obj, f"ridge_{feat_key}_s{seed}")
        model = Ridge(random_state=seed, **best_params)

    elif est_key == 'elasticnet':
        def obj(trial):
            m = ElasticNet(
                alpha=trial.suggest_float('alpha', 1e-5, 10.0, log=True),
                l1_ratio=trial.suggest_float('l1_ratio', 0.01, 0.99),
                max_iter=5000, random_state=seed)
            m.fit(X_train, y_train)
            return float(np.sqrt(np.mean((y_val - m.predict(X_val))**2)))
        best_params = _tune(obj, f"en_{feat_key}_s{seed}")
        model = ElasticNet(max_iter=5000, random_state=seed, **best_params)

    elif est_key == 'random_forest':
        def obj(trial):
            m = RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 15),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                n_jobs=-1, random_state=seed)
            m.fit(X_train, y_train)
            return float(np.sqrt(np.mean((y_val - m.predict(X_val))**2)))
        best_params = _tune(obj, f"rf_{feat_key}_s{seed}")
        model = RandomForestRegressor(n_jobs=-1, random_state=seed, **best_params)

    elif est_key == 'svr':
        def obj(trial):
            m = SVR(
                kernel='rbf',
                C=trial.suggest_float('C', 0.1, 100.0, log=True),
                gamma=trial.suggest_float('gamma', 1e-4, 1.0, log=True),
                epsilon=trial.suggest_float('epsilon', 0.001, 0.5, log=True),
                max_iter=10000)
            m.fit(X_train, y_train)
            return float(np.sqrt(np.mean((y_val - m.predict(X_val))**2)))
        best_params = _tune(obj, f"svr_{feat_key}_s{seed}")
        model = SVR(kernel='rbf', max_iter=10000, **best_params)

    else:
        raise ValueError(f"Unknown estimator: {est_key}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    result = {'feature_set': feat_key, 'estimator': est_key, 'family': family,
              'seed': seed, 'test_metrics': metrics, 'hpo': True,
              'best_params': best_params, 'needs_scale': needs_scale}
    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    with open(model_dir / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    logger.info(f"  [{feat_key} × {est_key}] seed={seed} R²={metrics['r2']:.4f} ρ={metrics['spearman']:.4f}")
    return result


def main():
    logger.info(f"Fast estimator comparison: {len(FEATURES)} features × {len(ESTIMATORS)} estimators × {len(SEEDS)} seeds")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for feat_key in FEATURES:
        safe_f = feat_key.replace(' ','_').replace('+','_')
        for est_key in ESTIMATORS:
            for seed in SEEDS:
                fpath = OUTPUT_DIR / f"{safe_f}_{est_key}" / f"seed_{seed}" / "metrics.json"
                if fpath.exists():
                    all_results.append(json.load(open(fpath)))
                    continue
                try:
                    all_results.append(run_one(est_key, feat_key, seed))
                except Exception as e:
                    logger.error(f"  [{feat_key} × {est_key}] seed={seed} FAILED: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    groups = defaultdict(list)
    for r in all_results:
        groups[(r['feature_set'], r['estimator'])].append(r)

    labels = {k: v[0] for k, v in ESTIMATORS.items()}
    families = {k: v[1] for k, v in ESTIMATORS.items()}

    feature_order = ['AA Comp', 'Chem', 'Site', 'Context', 'Site+Context', 'Chem+Site+Context']
    n_feat = len(feature_order)
    col_w = 13

    print("\n" + "=" * (26 + col_w * n_feat + 15))
    print("ESTIMATOR COMPARISON — ALL Optuna HPO, 10 trials each, 5 seeds")
    print("=" * (26 + col_w * n_feat + 15))
    header = f"{'Estimator':<18} {'Family':<8}"
    for f in feature_order:
        header += f" {f:>{col_w}}"
    header += f"  {'Trend':>12}"
    print(header)
    print("-" * (26 + col_w * n_feat + 15))

    for est in ESTIMATORS:
        row = f"{labels[est]:<18} {families[est]:<8}"
        all_vals = []
        for feat in feature_order:
            v = groups.get((feat, est), [])
            if v:
                r2s = [x['test_metrics']['r2'] for x in v]
                all_vals.append(np.mean(r2s))
                row += f" {np.mean(r2s):.4f}±{np.std(r2s,ddof=1):.4f}" if len(r2s)>1 else f" {np.mean(r2s):.4f}       "
            else:
                all_vals.append(float('nan'))
                row += f" {'...':>{col_w}}"
        # Trend check: Chem+Site+Context > Chem > AA Comp
        vs_seq  = all_vals[0] if len(all_vals)>0 else np.nan
        vs_chem = all_vals[1] if len(all_vals)>1 else np.nan
        vs_full = all_vals[5] if len(all_vals)>5 else np.nan
        trend = "✅ CONSISTENT" if (not np.isnan(vs_full) and not np.isnan(vs_chem) and not np.isnan(vs_seq) and vs_full > vs_chem > vs_seq) else ("⚠️DIFFERENT" if not np.isnan(vs_full) else "...")
        row += f"  {trend:>12}"
        print(row)

    print("-" * (26 + col_w * n_feat + 15))
    print("Trend = Chem+Site+Context > Chem > AA Comp (expected if site features add value universally)")
    # Show B/C without A results
    xgb_site    = groups.get(('Site', 'xgboost'), [])
    xgb_context = groups.get(('Context', 'xgboost'), [])
    xgb_sc      = groups.get(('Site+Context', 'xgboost'), [])
    if xgb_site:
        print(f"  Site without Chem (XGBoost): {np.mean([x['test_metrics']['r2'] for x in xgb_site]):.4f}")
    if xgb_context:
        print(f"  Context without Chem (XGBoost): {np.mean([x['test_metrics']['r2'] for x in xgb_context]):.4f}")
    if xgb_sc:
        print(f"  Site+Context without Chem (XGBoost): {np.mean([x['test_metrics']['r2'] for x in xgb_sc]):.4f}")

    # Save CSV with all 6 feature sets
    with open(OUTPUT_DIR / 'comparison_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        header = ['estimator','family']
        for feat in feature_order:
            header.extend([f'{feat}_R2', f'{feat}_R2_sd'])
        header.append('trend_consistent')
        w.writerow(header)
        for est in ESTIMATORS:
            row = [est, families[est]]
            for feat in feature_order:
                v = groups.get((feat, est), [])
                row.extend([np.mean([x['test_metrics']['r2'] for x in v]) if v else float('nan'),
                            np.std([x['test_metrics']['r2'] for x in v], ddof=1) if len(v)>1 else 0])
            vs_seq  = np.mean([x['test_metrics']['r2'] for x in groups.get(('AA Comp', est), [])]) if groups.get(('AA Comp', est)) else np.nan
            vs_chem = np.mean([x['test_metrics']['r2'] for x in groups.get(('Chem', est), [])]) if groups.get(('Chem', est)) else np.nan
            vs_full = np.mean([x['test_metrics']['r2'] for x in groups.get(('Chem+Site+Context', est), [])]) if groups.get(('Chem+Site+Context', est)) else np.nan
            trend_ok = not np.isnan(vs_full) and not np.isnan(vs_chem) and not np.isnan(vs_seq) and vs_full > vs_chem > vs_seq
            w.writerow(row + [trend_ok])

    logger.info(f"\nSummary: {OUTPUT_DIR / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()
