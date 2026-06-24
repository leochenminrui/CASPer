"""
Attachment-aware chemical representation for PEM.

Implements Tagged Chemical String (TCS) and Attachment-Aware Fingerprint (AAF)
representations with conservative scope (v1.0).

Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import re


class ParserStatus(str, Enum):
    """Parser status values."""
    SUCCESS = "success"
    INFERRED = "inferred"
    PARTIAL = "partial"
    FAILED = "failed"


class ExclusionReason(str, Enum):
    """Standardized exclusion reasons."""
    AMBIGUOUS_STRUCTURE = "ambiguous_structure"
    MISSING_ATTACHMENT = "missing_attachment"
    INVALID_POSITION = "invalid_position"
    UNSUPPORTED_MODIFICATION = "unsupported_modification"
    COMPLEX_PATTERN = "complex_pattern"
    INCOMPLETE_DATA = "incomplete_data"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class Attachment:
    """Attachment point specification."""
    position: int                # 0-indexed position in sequence
    residue: str                 # Single-letter amino acid code
    atom: str                    # Atom name (CA, NZ, OG, SG, etc.)
    bond: str                    # Bond type (single, double, triple)

    def to_tcs(self) -> str:
        """Convert to TCS attachment notation."""
        return f"{self.residue}{self.position}.{self.atom}.{self.bond}"

    def validate(self, sequence: str) -> bool:
        """Validate attachment against sequence."""
        if self.position < 0 or self.position >= len(sequence):
            return False
        if sequence[self.position] != self.residue:
            return False
        return True


@dataclass
class BridgeAttachment:
    """Bridge attachment (for cyclization)."""
    attach1: Attachment
    attach2: Attachment

    def to_tcs(self) -> str:
        """Convert to TCS bridge notation."""
        return f"{self.attach1.to_tcs()}-{self.attach2.to_tcs()}"

    def validate(self, sequence: str) -> bool:
        """Validate both attachment points."""
        return self.attach1.validate(sequence) and self.attach2.validate(sequence)


@dataclass
class SequenceContext:
    """Sequence context around anchor."""
    before: str                  # Upstream residues
    anchor: str                  # Anchor residue
    after: str                   # Downstream residues
    window: int = 3              # Context window size

    def to_tcs(self) -> str:
        """Convert to TCS context notation."""
        return f"[{self.before}-{self.anchor}-{self.after}]"

    @classmethod
    def from_sequence(
        cls,
        sequence: str,
        position: int,
        window: int = 3
    ) -> 'SequenceContext':
        """Extract context from sequence."""
        start = max(0, position - window)
        end = min(len(sequence), position + window + 1)

        before = sequence[start:position] if position > 0 else ""
        anchor = sequence[position]
        after = sequence[position + 1:end] if position < len(sequence) - 1 else ""

        # Pad if at boundaries
        if len(before) < window:
            before = "-" * (window - len(before)) + before
        if len(after) < window:
            after = after + "-" * (window - len(after))

        return cls(
            before=before,
            anchor=anchor,
            after=after,
            window=window
        )


@dataclass
class ChemRepr:
    """
    Complete chemical representation with attachment awareness.

    Supports both Tagged Chemical String (TCS) and
    Attachment-Aware Fingerprint (AAF) formats.
    """
    # Edit identification
    edit_id: str
    edit_family: str
    edit_type: str

    # Moiety
    moiety_smiles: str
    moiety_name: str

    # Attachment
    attachment: Attachment
    context: SequenceContext

    # Parsing metadata (no defaults — must precede fields with defaults)
    rule_id: str
    parser_status: ParserStatus
    canonicalization_notes: str

    # Provenance
    input_raw: str

    # Optional bridge (for cyclization)
    bridge_attachment: Optional[BridgeAttachment] = None
    bridge_context: Optional[SequenceContext] = None

    # Optional parsing metadata
    exclusion_reason: Optional[ExclusionReason] = None

    # Optional provenance
    parser_version: str = "1.0.0"

    # Cached representations
    _tcs: Optional[str] = field(default=None, repr=False)
    _aaf: Optional[List[float]] = field(default=None, repr=False)

    def to_tcs(self) -> str:
        """
        Generate Tagged Chemical String representation.

        Format: <moiety>@<attachment>:[context]
        """
        if self._tcs is not None:
            return self._tcs

        # Moiety
        moiety = self.moiety_name if self.moiety_name else self.moiety_smiles

        # Attachment
        if self.bridge_attachment:
            # Bridge format
            attach_str = self.bridge_attachment.to_tcs()
            context_str = f"{self.context.to_tcs()}~{self.bridge_context.to_tcs()}"
        else:
            # Single attachment
            attach_str = self.attachment.to_tcs()
            context_str = self.context.to_tcs()

        self._tcs = f"{moiety}@{attach_str}:{context_str}"
        return self._tcs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "edit_id": self.edit_id,
            "edit_family": self.edit_family,
            "edit_type": self.edit_type,
            "chem_repr": {
                "tcs": self.to_tcs(),
                "moiety_smiles": self.moiety_smiles,
                "moiety_name": self.moiety_name,
                "attachment": {
                    "position": self.attachment.position,
                    "residue": self.attachment.residue,
                    "atom": self.attachment.atom,
                    "bond": self.attachment.bond
                },
                "context": {
                    "before": self.context.before,
                    "anchor": self.context.anchor,
                    "after": self.context.after,
                    "window": self.context.window
                }
            },
            "parsing": {
                "rule_id": self.rule_id,
                "parser_status": self.parser_status.value,
                "canonicalization_notes": self.canonicalization_notes,
                "exclusion_reason": self.exclusion_reason.value if self.exclusion_reason else None
            },
            "provenance": {
                "input_raw": self.input_raw,
                "parser_version": self.parser_version
            }
        }

        # Add bridge info if present
        if self.bridge_attachment:
            result["chem_repr"]["bridge_attachment"] = {
                "position1": self.bridge_attachment.attach1.position,
                "residue1": self.bridge_attachment.attach1.residue,
                "atom1": self.bridge_attachment.attach1.atom,
                "bond1": self.bridge_attachment.attach1.bond,
                "position2": self.bridge_attachment.attach2.position,
                "residue2": self.bridge_attachment.attach2.residue,
                "atom2": self.bridge_attachment.attach2.atom,
                "bond2": self.bridge_attachment.attach2.bond
            }
            result["chem_repr"]["bridge_context"] = {
                "before": self.bridge_context.before,
                "anchor": self.bridge_context.anchor,
                "after": self.bridge_context.after,
                "window": self.bridge_context.window
            }

        return result

    def validate(self, sequence: str) -> Tuple[bool, Optional[str]]:
        """
        Validate representation against sequence.

        Returns:
            (is_valid, error_message)
        """
        # Validate primary attachment
        if not self.attachment.validate(sequence):
            return False, f"Invalid attachment at position {self.attachment.position}"

        # Validate context matches sequence
        actual_context = SequenceContext.from_sequence(
            sequence,
            self.attachment.position,
            self.context.window
        )

        if actual_context.anchor != self.context.anchor:
            return False, f"Context mismatch: expected {self.context.anchor}, got {actual_context.anchor}"

        # Validate bridge if present
        if self.bridge_attachment:
            if not self.bridge_attachment.validate(sequence):
                return False, "Invalid bridge attachment"

            actual_bridge_context = SequenceContext.from_sequence(
                sequence,
                self.bridge_attachment.attach2.position,
                self.bridge_context.window
            )

            if actual_bridge_context.anchor != self.bridge_context.anchor:
                return False, "Bridge context mismatch"

        return True, None


# Atom types valid for each residue (simplified)
RESIDUE_ATOMS = {
    'A': ['CA', 'CB'],
    'C': ['CA', 'CB', 'SG'],
    'D': ['CA', 'CB', 'CG', 'OD1', 'OD2'],
    'E': ['CA', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'F': ['CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'G': ['CA'],
    'H': ['CA', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'I': ['CA', 'CB', 'CG1', 'CG2', 'CD1'],
    'K': ['CA', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'L': ['CA', 'CB', 'CG', 'CD1', 'CD2'],
    'M': ['CA', 'CB', 'CG', 'SD', 'CE'],
    'N': ['CA', 'CB', 'CG', 'OD1', 'ND2'],
    'P': ['CA', 'CB', 'CG', 'CD'],
    'Q': ['CA', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'R': ['CA', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'S': ['CA', 'CB', 'OG'],
    'T': ['CA', 'CB', 'OG1', 'CG2'],
    'V': ['CA', 'CB', 'CG1', 'CG2'],
    'W': ['CA', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'Y': ['CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']
}


def validate_attachment_chemistry(residue: str, atom: str) -> bool:
    """
    Validate that atom type is valid for residue.

    Args:
        residue: Single-letter amino acid code
        atom: Atom name

    Returns:
        True if valid combination
    """
    if residue not in RESIDUE_ATOMS:
        return False
    return atom in RESIDUE_ATOMS[residue]
