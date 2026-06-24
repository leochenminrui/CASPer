"""
Base classes for baseline models.

Defines abstract interfaces and common functionality for all baselines.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
from datetime import datetime

from data.pem_schema import PEMSample


@dataclass
class BaselineConfig:
    """
    Configuration for a baseline model.

    Stores all hyperparameters and settings for reproducibility.
    """

    # Model identity
    name: str
    category: str  # "sequence_only", "chemistry_aware", "paired"
    version: str = "1.0.0"

    # Model hyperparameters
    model_type: str = "xgboost"  # xgboost, random_forest, mlp
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Featurization
    featurizer: str = "composition"  # composition, lm_embedding, descriptors, etc.
    featurizer_params: Dict[str, Any] = field(default_factory=dict)

    # Training
    random_seed: int = 42
    n_seeds: int = 5  # Number of random seeds for confidence intervals
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10

    # Validation
    validation_metric: str = "rmse"  # rmse, mae, r2, spearman
    minimize_metric: bool = True  # True for RMSE/MAE, False for R2/Spearman

    # Information matching
    use_edit_count: bool = False  # Whether to include edit count as feature
    use_edit_families: bool = False  # Whether to include edit family indicators

    # Hyperparameter tuning
    tune_hyperparameters: bool = False  # Whether to tune hyperparameters
    tuning_trials: int = 50  # Number of tuning trials

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'category': self.category,
            'version': self.version,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'featurizer': self.featurizer,
            'featurizer_params': self.featurizer_params,
            'random_seed': self.random_seed,
            'n_seeds': self.n_seeds,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'validation_metric': self.validation_metric,
            'minimize_metric': self.minimize_metric,
            'use_edit_count': self.use_edit_count,
            'use_edit_families': self.use_edit_families,
            'tune_hyperparameters': self.tune_hyperparameters,
            'tuning_trials': self.tuning_trials,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaselineConfig':
        """Create from dictionary."""
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save configuration to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'BaselineConfig':
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class BaselineResult:
    """
    Results from baseline model evaluation.

    Includes predictions, metrics, and statistical measures.
    """

    # Predictions
    predictions: np.ndarray
    targets: np.ndarray
    sample_ids: List[str]

    # Metrics (single seed)
    metrics: Dict[str, float]

    # Multi-seed statistics (if available)
    metrics_mean: Optional[Dict[str, float]] = None
    metrics_std: Optional[Dict[str, float]] = None
    metrics_ci_lower: Optional[Dict[str, float]] = None  # 95% CI lower
    metrics_ci_upper: Optional[Dict[str, float]] = None  # 95% CI upper

    # Model info
    model_name: str = ""
    split_name: str = ""  # train, val, test
    dataset: str = ""
    random_seed: int = 42

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'predictions': self.predictions.tolist(),
            'targets': self.targets.tolist(),
            'sample_ids': self.sample_ids,
            'metrics': self.metrics,
            'model_name': self.model_name,
            'split_name': self.split_name,
            'dataset': self.dataset,
            'random_seed': self.random_seed,
            'timestamp': self.timestamp,
        }

        if self.metrics_mean is not None:
            result['metrics_mean'] = self.metrics_mean
            result['metrics_std'] = self.metrics_std
            result['metrics_ci_lower'] = self.metrics_ci_lower
            result['metrics_ci_upper'] = self.metrics_ci_upper

        return result

    def save(self, path: Path) -> None:
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'BaselineResult':
        """Load results from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Convert lists back to arrays
        data['predictions'] = np.array(data['predictions'])
        data['targets'] = np.array(data['targets'])

        return cls(**data)


class BaselineModel(ABC):
    """
    Abstract base class for baseline models.

    All baselines implement this interface for consistent training and evaluation.
    """

    def __init__(self, config: BaselineConfig):
        """
        Initialize baseline model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.is_trained = False

    @abstractmethod
    def featurize(self, samples: List[PEMSample]) -> np.ndarray:
        """
        Convert samples to feature vectors.

        Args:
            samples: List of PEMSample objects

        Returns:
            Feature matrix (n_samples, n_features)
        """
        pass

    @abstractmethod
    def train(
        self,
        train_samples: List[PEMSample],
        val_samples: List[PEMSample],
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_samples: Training samples
            val_samples: Validation samples

        Returns:
            Training history/metrics
        """
        pass

    @abstractmethod
    def predict(self, samples: List[PEMSample]) -> np.ndarray:
        """
        Make predictions.

        Args:
            samples: Samples to predict

        Returns:
            Predictions (n_samples,)
        """
        pass

    def evaluate(
        self,
        samples: List[PEMSample],
        split_name: str = "test",
    ) -> BaselineResult:
        """
        Evaluate model on samples.

        Args:
            samples: Samples to evaluate
            split_name: Name of split (train/val/test)

        Returns:
            BaselineResult with predictions and metrics
        """
        # Get predictions
        predictions = self.predict(samples)

        # Extract targets
        targets = np.array([s.label for s in samples])
        sample_ids = [s.sample_id for s in samples]

        # Compute metrics
        metrics = self._compute_metrics(predictions, targets)

        return BaselineResult(
            predictions=predictions,
            targets=targets,
            sample_ids=sample_ids,
            metrics=metrics,
            model_name=self.config.name,
            split_name=split_name,
            dataset=samples[0].dataset if samples else "",
            random_seed=self.config.random_seed,
        )

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth labels

        Returns:
            Dictionary of metrics
        """
        from scipy.stats import spearmanr
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(targets, predictions))),
            'mae': float(mean_absolute_error(targets, predictions)),
            'r2': float(r2_score(targets, predictions)),
            'spearman': float(spearmanr(targets, predictions)[0]),
        }

        return metrics

    def save_model(self, path: Path) -> None:
        """
        Save trained model.

        Args:
            path: Path to save model
        """
        import pickle

        if not self.is_trained:
            raise ValueError("Model not trained yet")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save config
        config_path = path.parent / f"{path.stem}_config.json"
        self.config.save(config_path)

    def load_model(self, path: Path) -> None:
        """
        Load trained model.

        Args:
            path: Path to model file
        """
        import pickle

        path = Path(path)

        # Load model
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

        # Load config if available
        config_path = path.parent / f"{path.stem}_config.json"
        if config_path.exists():
            self.config = BaselineConfig.load(config_path)

        self.is_trained = True

    def get_feature_names(self) -> List[str]:
        """
        Get names of features.

        Returns:
            List of feature names
        """
        return []  # Override in subclasses

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance (if available).

        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        return None  # Override in subclasses if applicable


def train_with_multiple_seeds(
    model_class,
    config: BaselineConfig,
    train_samples: List[PEMSample],
    val_samples: List[PEMSample],
    test_samples: List[PEMSample],
    n_seeds: Optional[int] = None,
) -> Tuple[List[BaselineResult], BaselineResult]:
    """
    Train model with multiple random seeds and compute statistics.

    Args:
        model_class: Baseline model class
        config: Base configuration
        train_samples: Training samples
        val_samples: Validation samples
        test_samples: Test samples
        n_seeds: Number of seeds (default: use config.n_seeds)

    Returns:
        (list of per-seed results, aggregated result with statistics)
    """
    n_seeds = n_seeds or config.n_seeds
    base_seed = config.random_seed

    all_results = []

    for i in range(n_seeds):
        # Create config with different seed
        seed_config = BaselineConfig(**{
            **config.to_dict(),
            'random_seed': base_seed + i,
        })

        # Train model
        model = model_class(seed_config)
        model.train(train_samples, val_samples)

        # Evaluate on test
        result = model.evaluate(test_samples, split_name="test")
        all_results.append(result)

    # Aggregate statistics
    aggregated = _aggregate_results(all_results, config.name, "test")

    return all_results, aggregated


def _aggregate_results(
    results: List[BaselineResult],
    model_name: str,
    split_name: str,
) -> BaselineResult:
    """
    Aggregate results from multiple seeds.

    Computes mean, std, and 95% confidence intervals.

    Args:
        results: List of results from different seeds
        model_name: Name of model
        split_name: Name of split

    Returns:
        Aggregated result with statistics
    """
    # Extract metrics from all results
    metric_names = list(results[0].metrics.keys())
    metric_values = {name: [] for name in metric_names}

    for result in results:
        for name in metric_names:
            metric_values[name].append(result.metrics[name])

    # Compute statistics
    metrics_mean = {
        name: float(np.mean(values))
        for name, values in metric_values.items()
    }

    metrics_std = {
        name: float(np.std(values, ddof=1))
        for name, values in metric_values.items()
    }

    # 95% confidence intervals (t-distribution)
    from scipy import stats as sp_stats
    n = len(results)
    t_value = sp_stats.t.ppf(0.975, n - 1)  # 95% CI

    metrics_ci_lower = {
        name: metrics_mean[name] - t_value * metrics_std[name] / np.sqrt(n)
        for name in metric_names
    }

    metrics_ci_upper = {
        name: metrics_mean[name] + t_value * metrics_std[name] / np.sqrt(n)
        for name in metric_names
    }

    # Use first result's predictions as representative
    return BaselineResult(
        predictions=results[0].predictions,
        targets=results[0].targets,
        sample_ids=results[0].sample_ids,
        metrics=metrics_mean,  # Use mean as primary metrics
        metrics_mean=metrics_mean,
        metrics_std=metrics_std,
        metrics_ci_lower=metrics_ci_lower,
        metrics_ci_upper=metrics_ci_upper,
        model_name=model_name,
        split_name=split_name,
        dataset=results[0].dataset,
        random_seed=results[0].random_seed,
    )
