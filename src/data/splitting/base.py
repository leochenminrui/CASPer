"""
Base classes for data splitting strategies.

Defines abstract interfaces and data structures for implementing
and evaluating different split strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from ..pem_schema import PEMSample


class LeakageRisk(str, Enum):
    """Classification of data leakage risk."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class LeakageAnalysis:
    """Results of leakage analysis for a split."""

    # Sequence similarity metrics
    max_sequence_identity: float = 0.0
    pct_test_high_similarity: float = 0.0  # % of test with >70% identity to train
    mean_test_train_similarity: float = 0.0

    # Scaffold overlap metrics
    n_shared_scaffolds: int = 0
    pct_test_shared_scaffold: float = 0.0

    # Edit pattern similarity
    edit_profile_cosine_similarity: float = 0.0
    edit_family_js_divergence: float = 0.0

    # Label distribution
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    train_label_mean: float = 0.0
    train_label_std: float = 0.0
    val_label_mean: float = 0.0
    val_label_std: float = 0.0
    test_label_mean: float = 0.0
    test_label_std: float = 0.0

    # Overall risk
    overall_risk: LeakageRisk = LeakageRisk.HIGH

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sequence_similarity': {
                'max_identity': self.max_sequence_identity,
                'pct_high_similarity': self.pct_test_high_similarity,
                'mean_test_train': self.mean_test_train_similarity,
            },
            'scaffold_overlap': {
                'n_shared': self.n_shared_scaffolds,
                'pct_test_shared': self.pct_test_shared_scaffold,
            },
            'edit_pattern_similarity': {
                'cosine_similarity': self.edit_profile_cosine_similarity,
                'js_divergence': self.edit_family_js_divergence,
            },
            'label_distribution': {
                'ks_statistic': self.ks_statistic,
                'ks_pvalue': self.ks_pvalue,
                'train': {
                    'mean': self.train_label_mean,
                    'std': self.train_label_std,
                },
                'val': {
                    'mean': self.val_label_mean,
                    'std': self.val_label_std,
                },
                'test': {
                    'mean': self.test_label_mean,
                    'std': self.test_label_std,
                },
            },
            'overall_risk': self.overall_risk,
        }


@dataclass
class SplitMetadata:
    """Metadata for a split strategy execution."""

    strategy_name: str
    feasible: bool
    reason: Optional[str] = None  # If not feasible, why?

    # Counts
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0

    # Strategy-specific info
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    n_scaffolds: Optional[int] = None
    n_clusters: Optional[int] = None
    n_edit_profiles: Optional[int] = None
    n_derivative_groups: Optional[int] = None

    # Distributions
    scaffold_distribution: Optional[Dict[str, int]] = None
    cluster_distribution: Optional[Dict[str, int]] = None
    edit_profile_distribution: Optional[Dict[str, int]] = None

    # Leakage analysis
    leakage_analysis: Optional[LeakageAnalysis] = None

    # Provenance
    random_seed: int = 42
    split_version: str = "1.0.0"
    creation_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'strategy_name': self.strategy_name,
            'feasible': self.feasible,
            'train_count': self.train_count,
            'val_count': self.val_count,
            'test_count': self.test_count,
            'strategy_params': self.strategy_params,
            'random_seed': self.random_seed,
            'split_version': self.split_version,
        }

        if self.reason:
            result['reason'] = self.reason

        if self.n_scaffolds is not None:
            result['n_scaffolds'] = self.n_scaffolds
        if self.n_clusters is not None:
            result['n_clusters'] = self.n_clusters
        if self.n_edit_profiles is not None:
            result['n_edit_profiles'] = self.n_edit_profiles
        if self.n_derivative_groups is not None:
            result['n_derivative_groups'] = self.n_derivative_groups

        if self.scaffold_distribution:
            result['scaffold_distribution'] = self.scaffold_distribution
        if self.cluster_distribution:
            result['cluster_distribution'] = self.cluster_distribution
        if self.edit_profile_distribution:
            result['edit_profile_distribution'] = self.edit_profile_distribution

        if self.leakage_analysis:
            result['leakage_analysis'] = self.leakage_analysis.to_dict()

        if self.creation_timestamp:
            result['creation_timestamp'] = self.creation_timestamp

        return result


@dataclass
class SplitResult:
    """Result of applying a split strategy."""

    train_samples: List[PEMSample]
    val_samples: List[PEMSample]
    test_samples: List[PEMSample]
    metadata: SplitMetadata

    def __post_init__(self):
        """Validate split result."""
        # Check no overlap
        train_ids = {s.sample_id for s in self.train_samples}
        val_ids = {s.sample_id for s in self.val_samples}
        test_ids = {s.sample_id for s in self.test_samples}

        overlap_train_val = train_ids & val_ids
        overlap_train_test = train_ids & test_ids
        overlap_val_test = val_ids & test_ids

        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise ValueError(
                f"Split has overlapping samples! "
                f"train∩val={len(overlap_train_val)}, "
                f"train∩test={len(overlap_train_test)}, "
                f"val∩test={len(overlap_val_test)}"
            )

        # Update counts in metadata
        self.metadata.train_count = len(self.train_samples)
        self.metadata.val_count = len(self.val_samples)
        self.metadata.test_count = len(self.test_samples)

    @property
    def total_samples(self) -> int:
        """Total number of samples across all splits."""
        return len(self.train_samples) + len(self.val_samples) + len(self.test_samples)

    def get_split_proportions(self) -> Tuple[float, float, float]:
        """Get actual train/val/test proportions."""
        total = self.total_samples
        if total == 0:
            return 0.0, 0.0, 0.0
        return (
            len(self.train_samples) / total,
            len(self.val_samples) / total,
            len(self.test_samples) / total,
        )


class SplitStrategy(ABC):
    """
    Abstract base class for split strategies.

    Each strategy implements a different approach to splitting data
    while avoiding specific types of leakage.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        """
        Initialize split strategy.

        Args:
            train_ratio: Proportion for training set (default 0.70)
            val_ratio: Proportion for validation set (default 0.15)
            test_ratio: Proportion for test set (default 0.15)
            random_seed: Random seed for reproducibility (default 42)
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    @abstractmethod
    def check_feasibility(self, samples: List[PEMSample]) -> Tuple[bool, Optional[str]]:
        """
        Check if this split strategy is feasible for the given samples.

        Args:
            samples: List of PEMSample objects

        Returns:
            (is_feasible, reason_if_not)
        """
        pass

    @abstractmethod
    def split(self, samples: List[PEMSample]) -> SplitResult:
        """
        Split samples into train/val/test sets.

        Args:
            samples: List of PEMSample objects

        Returns:
            SplitResult with train/val/test splits and metadata

        Raises:
            ValueError: If split is not feasible
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        pass

    def _validate_proportions(
        self,
        train_count: int,
        val_count: int,
        test_count: int,
        total_count: int,
    ) -> None:
        """
        Validate that actual proportions are within ±2% of target.

        Args:
            train_count: Number of training samples
            val_count: Number of validation samples
            test_count: Number of test samples
            total_count: Total number of samples

        Raises:
            ValueError: If proportions are too far from target
        """
        if total_count == 0:
            return

        actual_train = train_count / total_count
        actual_val = val_count / total_count
        actual_test = test_count / total_count

        tolerance = 0.02  # ±2%

        if abs(actual_train - self.train_ratio) > tolerance:
            raise ValueError(
                f"Train proportion {actual_train:.3f} too far from target "
                f"{self.train_ratio:.3f} (tolerance ±{tolerance})"
            )
        if abs(actual_val - self.val_ratio) > tolerance:
            raise ValueError(
                f"Val proportion {actual_val:.3f} too far from target "
                f"{self.val_ratio:.3f} (tolerance ±{tolerance})"
            )
        if abs(actual_test - self.test_ratio) > tolerance:
            raise ValueError(
                f"Test proportion {actual_test:.3f} too far from target "
                f"{self.test_ratio:.3f} (tolerance ±{tolerance})"
            )

    def _create_metadata(
        self,
        feasible: bool,
        reason: Optional[str] = None,
        **kwargs
    ) -> SplitMetadata:
        """
        Create split metadata.

        Args:
            feasible: Whether split was feasible
            reason: Reason if not feasible
            **kwargs: Additional metadata fields

        Returns:
            SplitMetadata object
        """
        from datetime import datetime

        # Build base strategy params, merging with any caller-provided overrides
        base_params = {
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
        }
        if 'strategy_params' in kwargs:
            base_params.update(kwargs.pop('strategy_params'))

        return SplitMetadata(
            strategy_name=self.get_strategy_name(),
            feasible=feasible,
            reason=reason,
            random_seed=self.random_seed,
            split_version="1.0.0",
            creation_timestamp=datetime.now().isoformat(),
            strategy_params=base_params,
            **kwargs
        )
