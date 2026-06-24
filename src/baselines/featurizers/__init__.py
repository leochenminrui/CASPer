"""
Featurizers for baseline models.

Converts PEMSample objects into feature vectors.
"""

from .composition import (
    AACompositionFeaturizer,
    compute_aa_composition,
)

from .descriptors import (
    ChemicalDescriptorFeaturizer,
    DescriptorOnlyFeaturizer,
)

from .edit_features import (
    EditCountFeaturizer,
    AnchorWindowFeaturizer,
)

__all__ = [
    # Composition
    'AACompositionFeaturizer',
    'compute_aa_composition',

    # Chemical descriptors
    'ChemicalDescriptorFeaturizer',
    'DescriptorOnlyFeaturizer',

    # Edit features
    'EditCountFeaturizer',
    'AnchorWindowFeaturizer',
]
