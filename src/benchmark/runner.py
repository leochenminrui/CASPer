"""
Benchmark runner — orchestrates model training, HPO, and evaluation.

Supports resume: if output files already exist for a (model, seed, split),
they are skipped unless --force is used.
"""

import json
import csv
import logging
import traceback
import warnings
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Ensure src/ is on path for sibling imports
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from .registry import MODEL_REGISTRY, ModelSpec, list_implemented_models
from .featurizers import FEATURIZER_REGISTRY
from .evaluation import compute_all_metrics
from .optuna_tuner import tune_xgboost, tune_random_forest, \
    tune_ridge, tune_elasticnet, tune_svr

logger = logging.getLogger(__name__)


def _load_samples(split_dir: Path, split_name: str) -> List:
    """Load PEM samples from a split directory."""
    from data.serialization import load_samples as pem_load
    jsonl_path = split_dir / f"{split_name}.jsonl"
    parquet_path = split_dir / f"{split_name}.parquet"
    if jsonl_path.exists():
        return pem_load(jsonl_path)
    elif parquet_path.exists():
        return pem_load(parquet_path)
    else:
        raise FileNotFoundError(f"No {split_name} file in {split_dir}")


def run_single_model(
    model_id: str,
    split_type: str,
    seed: int,
    config: Dict[str, Any],
    output_dir: Path,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Train and evaluate a single model for one seed/split combination.

    Args:
        model_id: Model identifier from MODEL_REGISTRY
        split_type: 'random' or 'sequence_cluster'
        seed: Random seed
        config: Full benchmark config dict
        output_dir: Base output directory
        resume: Skip if output files already exist

    Returns:
        Dict with metrics, predictions, status
    """
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        return {'status': 'failed', 'error': f'Unknown model_id: {model_id}'}

    model_dir = output_dir / split_type / f"seed_{seed}" / model_id
    metrics_file = model_dir / "metrics.json"
    preds_file = model_dir / "predictions.csv"

    # Resume check
    if resume and metrics_file.exists() and preds_file.exists():
        logger.info(f"  [{model_id}] seed={seed} — already completed, skipping")
        with open(metrics_file) as f:
            saved = json.load(f)
        return {'status': 'completed', **saved}

    model_dir.mkdir(parents=True, exist_ok=True)

    # Handle non-implemented models
    if spec.status == "requires_external":
        result = {
            'model_id': model_id,
            'seed': seed,
            'split_type': split_type,
            'status': 'not_run',
            'reason': f'Requires: {spec.requires_external}',
            'hpo': spec.hpo,
        }
        with open(metrics_file, 'w') as f:
            json.dump(result, f, indent=2)
        return result

    try:
        # ── 1. Load data ─────────────────────────────────────────────────
        from data.loader import load_pem_dataset
        dataset = "CycPeptMPDB_PAMPA"

        if split_type == "sequence_cluster":
            # Try loading cluster split; if unavailable, fall back
            try:
                data = load_pem_dataset(dataset, "sequence_cluster")
            except FileNotFoundError:
                logger.warning(
                    f"Sequence-cluster split not found for {dataset}. "
                    "Run scripts/create_sequence_cluster_split.py first.")
                result = {
                    'model_id': model_id, 'seed': seed,
                    'split_type': split_type, 'status': 'failed',
                    'error': 'Sequence-cluster split not available',
                }
                with open(metrics_file, 'w') as f:
                    json.dump(result, f, indent=2)
                return result
        else:
            data = load_pem_dataset(dataset, "random")

        train_samples = data['train']
        val_samples = data['val']
        test_samples = data['test']

        # ── 2. Featurize ─────────────────────────────────────────────────
        featurizer_cls = FEATURIZER_REGISTRY.get(spec.featurizer)
        if featurizer_cls is None:
            raise ValueError(f"Unknown featurizer: {spec.featurizer}")

        featurizer = featurizer_cls(**spec.featurizer_kwargs)
        featurizer.fit(train_samples)

        X_train = featurizer.transform(train_samples)
        X_val = featurizer.transform(val_samples)
        X_test = featurizer.transform(test_samples)

        y_train = np.array([s.label for s in train_samples])
        y_val = np.array([s.label for s in val_samples])
        y_test = np.array([s.label for s in test_samples])

        # Handle NaN in features (robustness)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        test_ids = [s.sample_id for s in test_samples]

        # ── 3. HPO & Train ───────────────────────────────────────────────
        xgb_cfg = config.get('xgboost', {})
        rf_cfg = config.get('random_forest', {})
        n_trials = config.get('benchmark', {}).get('n_trials', 50)
        optuna_cfg = config.get('optuna', {})

        hpo_used = False
        best_params = {}

        # Flags for scaling (linear/SVR models need it, tree models don't)
        needs_scale = spec.estimator_type in ('ridge', 'elasticnet', 'svr')
        scaler = None
        if needs_scale:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)  # keep scaled test ready
        else:
            X_test_s = X_test

        if spec.hpo and spec.estimator_type == "xgboost":
            hpo_used = True
            tune_result = tune_xgboost(
                X_train, y_train, X_val, y_val,
                n_trials=n_trials,
                random_seed=seed,
                output_dir=model_dir,
                search_space=xgb_cfg.get('search_space'),
                fixed_params=xgb_cfg.get('fixed_params', {}),
            )
            best_params = tune_result['best_params']
            model = tune_result.get('best_model')
            val_metrics = tune_result['best_val_metrics']

        elif spec.hpo and spec.estimator_type == "random_forest":
            hpo_used = True
            tune_result = tune_random_forest(
                X_train, y_train, X_val, y_val,
                n_trials=n_trials,
                random_seed=seed,
                output_dir=model_dir,
                search_space=rf_cfg.get('search_space'),
                fixed_params=rf_cfg.get('fixed_params', {}),
            )
            best_params = tune_result['best_params']
            model = tune_result.get('best_model')
            val_metrics = tune_result['best_val_metrics']


        elif spec.hpo and spec.estimator_type == "ridge":
            hpo_used = True
            tune_result = tune_ridge(
                X_train, y_train, X_val, y_val,
                n_trials=n_trials, random_seed=seed,
                output_dir=model_dir,
            )
            best_params = tune_result['best_params']
            model = tune_result.get('best_model')
            val_metrics = tune_result['best_val_metrics']

        elif spec.hpo and spec.estimator_type == "elasticnet":
            hpo_used = True
            tune_result = tune_elasticnet(
                X_train, y_train, X_val, y_val,
                n_trials=n_trials, random_seed=seed,
                output_dir=model_dir,
            )
            best_params = tune_result['best_params']
            model = tune_result.get('best_model')
            val_metrics = tune_result['best_val_metrics']

        elif spec.hpo and spec.estimator_type == "svr":
            hpo_used = True
            tune_result = tune_svr(
                X_train, y_train, X_val, y_val,
                n_trials=n_trials, random_seed=seed,
                output_dir=model_dir,
            )
            best_params = tune_result['best_params']
            model = tune_result.get('best_model')
            val_metrics = tune_result['best_val_metrics']

        else:
            # No HPO — use default XGBoost params
            from xgboost import XGBRegressor
            fixed = xgb_cfg.get('fixed_params', {})
            model = XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                verbosity=0,
                n_jobs=-1,
                random_state=seed,
            )
            model.fit(X_train, y_train, verbose=False)
            y_val_pred = model.predict(X_val)
            val_metrics = compute_all_metrics(y_val, y_val_pred)

        # ── 4. Test Evaluation ───────────────────────────────────────────
        y_test_pred = model.predict(X_test_s)
        test_metrics = compute_all_metrics(y_test, y_test_pred)

        # ── 5. Feature Importance (XGBoost only) ─────────────────────────
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            try:
                names = featurizer.get_feature_names()
                importances = model.feature_importances_
                feature_importance = dict(
                    sorted(zip(names, importances),
                           key=lambda x: x[1], reverse=True)[:30])
            except Exception:
                pass

        # ── 6. Save Results ──────────────────────────────────────────────
        result = {
            'model_id': model_id,
            'model_name': spec.model_name,
            'role': spec.role,
            'seed': seed,
            'split_type': split_type,
            'status': 'completed',
            'hpo': hpo_used,
            'n_features': int(X_train.shape[1]),
            'n_train': len(train_samples),
            'n_val': len(val_samples),
            'n_test': len(test_samples),
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'best_params': best_params,
            'feature_importance_top30': feature_importance,
        }

        with open(metrics_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        # Save predictions CSV
        with open(preds_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_id', 'y_true', 'y_pred', 'split', 'model_id',
                'seed', 'split_type'
            ])
            for sid, yt, yp in zip(test_ids, y_test, y_test_pred):
                writer.writerow([
                    sid, float(yt), float(yp), 'test', model_id,
                    seed, split_type
                ])

        logger.info(
            f"  [{model_id}] seed={seed} "
            f"R²={test_metrics['r2']:.4f} "
            f"RMSE={test_metrics['rmse']:.4f} "
            f"ρ={test_metrics['spearman']:.4f}")

        return result

    except Exception as e:
        logger.error(f"  [{model_id}] seed={seed} FAILED: {e}")
        traceback.print_exc()
        result = {
            'model_id': model_id,
            'seed': seed,
            'split_type': split_type,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
        }
        with open(metrics_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        return result


class BenchmarkRunner:
    """Orchestrates the full benchmark across models, splits, and seeds."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bench_cfg = config.get('benchmark', {})
        self.output_base = Path(config.get('output', {}).get(
            'base_dir', 'results/benchmark'))

    def run(self, models: Optional[List[str]] = None,
            splits: Optional[List[str]] = None,
            seeds: Optional[List[int]] = None,
            resume: bool = True):
        """
        Run the full benchmark.

        Args:
            models: Model IDs to run (None = all implemented)
            splits: Split types (None = from config)
            seeds: Random seeds (None = from config)
            resume: Skip completed runs
        """
        if models is None:
            include = self.bench_cfg.get('models', {}).get('include', [])
            exclude = set(self.bench_cfg.get('models', {}).get('exclude', []))
            if include:
                models = [m for m in include if m not in exclude]
            else:
                models = list_implemented_models()

        if splits is None:
            splits = self.bench_cfg.get('splits', ['random'])

        if seeds is None:
            seeds = self.bench_cfg.get('seeds', [0, 1, 2, 3, 4])

        n_trials = self.bench_cfg.get('n_trials', 50)

        logger.info("=" * 60)
        logger.info(f"CASPer Benchmark: {self.bench_cfg.get('name', '')}")
        logger.info(f"Models: {len(models)}, Splits: {splits}, "
                    f"Seeds: {seeds}, Trials: {n_trials}")
        logger.info("=" * 60)

        total = len(models) * len(splits) * len(seeds)
        completed = 0
        failed = 0
        skipped_external = 0

        for split_type in splits:
            for model_id in models:
                for seed in seeds:
                    result = run_single_model(
                        model_id=model_id,
                        split_type=split_type,
                        seed=seed,
                        config=self.config,
                        output_dir=self.output_base,
                        resume=resume,
                    )
                    status = result.get('status', 'unknown')
                    if status == 'completed':
                        completed += 1
                    elif status == 'failed':
                        failed += 1
                    elif status == 'not_run':
                        skipped_external += 1

        logger.info("=" * 60)
        logger.info(f"Benchmark complete: {completed} completed, "
                    f"{failed} failed, {skipped_external} skipped (external)")
        logger.info(f"Results: {self.output_base}")
        logger.info("=" * 60)
