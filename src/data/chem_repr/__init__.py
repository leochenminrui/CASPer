"""Attachment-aware chemical representation for PEM."""

from .representation import (
    ChemRepr,
    Attachment,
    BridgeAttachment,
    SequenceContext,
    ParserStatus,
    ExclusionReason
)

from .canonicalization import (
    canonicalize_edit,
    CANONICALIZATION_RULES
)

__all__ = [
    'ChemRepr',
    'Attachment',
    'BridgeAttachment',
    'SequenceContext',
    'ParserStatus',
    'ExclusionReason',
    'canonicalize_edit',
    'CANONICALIZATION_RULES'
]
