"""
Data splitting module for PEM.

Implements multiple splitting strategies with leakage analysis and
feasibility checking.
"""

from .base import (
    SplitStrategy,
    SplitResult,
    SplitMetadata,
    LeakageAnalysis,
)

from .strategies import (
    RandomSplitter,
    ScaffoldAwareSplitter,
    SequenceClusterSplitter,
    EditFamilyAwareSplitter,
    SameEditFamilyNewScaffoldSplitter,
    GroupedDerivativeSplitter,
)

from .utils import (
    extract_scaffold,
    cluster_sequences,
    compute_sequence_similarity,
    identify_derivative_groups,
)

from .leakage import (
    analyze_leakage,
    compute_sequence_similarity_matrix,
    check_scaffold_overlap,
    compare_label_distributions,
)

__all__ = [
    # Base classes
    'SplitStrategy',
    'SplitResult',
    'SplitMetadata',
    'LeakageAnalysis',

    # Strategies
    'RandomSplitter',
    'ScaffoldAwareSplitter',
    'SequenceClusterSplitter',
    'EditFamilyAwareSplitter',
    'SameEditFamilyNewScaffoldSplitter',
    'GroupedDerivativeSplitter',

    # Utils
    'extract_scaffold',
    'cluster_sequences',
    'compute_sequence_similarity',
    'identify_derivative_groups',

    # Leakage analysis
    'analyze_leakage',
    'compute_sequence_similarity_matrix',
    'check_scaffold_overlap',
    'compare_label_distributions',
]
