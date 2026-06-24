"""
Baseline models for PEM benchmark.

Implements information-matched baselines:
- Sequence-only baselines (composition)
- Chemistry-aware baselines (descriptor-only)

All baselines use the same data splits and evaluation protocols for fair comparison.
"""

from .base import BaselineModel, BaselineConfig, BaselineResult

__all__ = [
    'BaselineModel',
    'BaselineConfig',
    'BaselineResult',
]
