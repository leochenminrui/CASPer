"""
Anchor-Aware Descriptor Baseline.

Uses anchor-aware descriptors with XGBoost for permeability prediction.
"""

from typing import List, Dict, Any
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from baselines.base import BaselineModel, BaselineConfig
from baselines.featurizers.anchor_aware_descriptors import AnchorAwareDescriptorFeaturizer
from data.pem_schema import PEMSample


class AnchorAwareDescriptorBaseline(BaselineModel):
    """
    Anchor-aware descriptor baseline.

    Uses anchor-aware chemical descriptors + position/residue features.
    """

    def __init__(self, config: BaselineConfig):
        """
        Initialize anchor-aware descriptor baseline.

        Args:
            config: Model configuration
        """
        super().__init__(config)

        # Initialize featurizer
        self.featurizer = AnchorAwareDescriptorFeaturizer(
            descriptor_set=config.featurizer_params.get('descriptor_set', 'basic'),
            ablation_mode=config.featurizer_params.get('ablation_mode', 'full'),
        )

        # Initialize model
        if config.model_type == "xgboost":
            self.model = XGBRegressor(
                random_state=config.random_seed,
                **config.model_params,
            )
        elif config.model_type == "random_forest":
            self.model = RandomForestRegressor(
                random_state=config.random_seed,
                **config.model_params,
            )
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

    def featurize(self, samples: List[PEMSample]) -> np.ndarray:
        """
        Convert samples to feature vectors.

        Args:
            samples: List of PEMSample objects

        Returns:
            Feature matrix (n_samples, 73)
        """
        return self.featurizer.featurize(samples)

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
            Training history
        """
        # Featurize
        X_train = self.featurize(train_samples)
        y_train = np.array([s.label for s in train_samples])

        X_val = self.featurize(val_samples)
        y_val = np.array([s.label for s in val_samples])

        # Train
        if isinstance(self.model, XGBRegressor):
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:  # Random Forest
            self.model.fit(X_train, y_train)

        self.is_trained = True

        # Validation metrics
        val_pred = self.model.predict(X_val)
        val_metrics = self._compute_metrics(val_pred, y_val)

        return {'val_metrics': val_metrics}

    def predict(self, samples: List[PEMSample]) -> np.ndarray:
        """
        Make predictions.

        Args:
            samples: Samples to predict

        Returns:
            Predictions (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X = self.featurize(samples)
        return self.model.predict(X)

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.featurizer.get_feature_names()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_trained:
            return {}

        if hasattr(self.model, 'feature_importances_'):
            names = self.get_feature_names()
            importances = self.model.feature_importances_
            return dict(zip(names, importances))

        return {}
