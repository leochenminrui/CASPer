"""
CASPer Benchmark Framework — Core Package.

Unified benchmark pipeline for fair comparison of all project models:
internal baselines, primary paper models, ablation controls, and generic chemistry benchmarks.

All models use identical data splits, metrics, and Optuna HPO.
"""

from .registry import MODEL_REGISTRY, get_model_config, list_models_by_role
from .runner import BenchmarkRunner, run_single_model
from .evaluation import compute_all_metrics
from .optuna_tuner import tune_xgboost, tune_random_forest

__all__ = [
    'MODEL_REGISTRY',
    'get_model_config',
    'list_models_by_role',
    'BenchmarkRunner',
    'run_single_model',
    'compute_all_metrics',
    'tune_xgboost',
    'tune_random_forest',
]
