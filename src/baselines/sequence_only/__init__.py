"""
Sequence-only baseline models.

These baselines use only sequence information (no chemical modification data).
"""

from .composition_baseline import CompositionBaseline

__all__ = [
    'CompositionBaseline',
]
