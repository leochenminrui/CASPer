"""Dataset-to-PEM schema converters.

This study uses a single dataset: CycPeptMPDB (PAMPA subset).
Skeleton converters for PepMSND and DBAASP were removed before submission
to avoid repository–manuscript inconsistency; they are archived in
archived/future_extension/ for reference.
"""

from .base_converter import BaseConverter
from .cycpeptmpdb_converter import CycPeptMPDBConverter

__all__ = [
    'BaseConverter',
    'CycPeptMPDBConverter',
]
