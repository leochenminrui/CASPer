"""
Chemistry-aware baseline models.

These baselines use chemical modification information but without learnable anchors.
"""

from .descriptor_only import DescriptorOnlyBaseline

__all__ = [
    'DescriptorOnlyBaseline',
]
