"""
Standardized data schema for PEM project.

Defines the common format that all datasets will be converted to after processing.
This ensures consistent handling across different source formats.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum


class ModificationType(Enum):
    """Types of chemical modifications."""
    N_TERMINAL = "n_terminal"
    C_TERMINAL = "c_terminal"
    SIDE_CHAIN = "side_chain"
    BACKBONE = "backbone"
    CYCLIZATION = "cyclization"
    D_AMINO_ACID = "d_amino_acid"
    OTHER = "other"


@dataclass
class Modification:
    """
    Represents a single chemical modification to a peptide.

    Attributes:
        position: Residue position (0-indexed), None for terminal modifications
        modification_type: Type of modification
        modification_name: Standard name (e.g., 'acetylation', 'phosphorylation')
        notation: Original notation from source dataset
        details: Additional modification-specific information
    """
    modification_type: ModificationType
    modification_name: str
    notation: str
    position: Optional[int] = None
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate modification."""
        if self.modification_type == ModificationType.SIDE_CHAIN:
            assert self.position is not None, "Side chain modification requires position"


@dataclass
class Peptide:
    """
    Standardized peptide representation.

    This is the core data structure that all datasets are converted to.

    Attributes:
        sequence: Amino acid sequence (single letter code, canonical residues)
        modifications: List of chemical modifications
        cyclization_info: Information about cyclization if applicable
        original_notation: Original sequence notation from source
        metadata: Additional peptide-level metadata
    """
    sequence: str
    modifications: List[Modification] = field(default_factory=list)
    cyclization_info: Optional[Dict[str, Any]] = None
    original_notation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate peptide structure."""
        # Check sequence contains only standard amino acids
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        assert all(aa in valid_aa for aa in self.sequence), \
            f"Sequence contains non-standard amino acids: {self.sequence}"

        # Validate modification positions
        for mod in self.modifications:
            if mod.position is not None:
                assert 0 <= mod.position < len(self.sequence), \
                    f"Modification position {mod.position} out of range for sequence length {len(self.sequence)}"

    def get_modified_positions(self) -> List[int]:
        """Get all positions with modifications."""
        return sorted([m.position for m in self.modifications if m.position is not None])

    def is_cyclic(self) -> bool:
        """Check if peptide is cyclic."""
        return self.cyclization_info is not None


@dataclass
class DataPoint:
    """
    Complete data point for modeling.

    Combines peptide structure with experimental measurement and metadata.

    Attributes:
        id: Unique identifier
        dataset: Source dataset name
        peptide: Peptide structure
        property_name: Name of measured property
        property_value: Measured value
        property_unit: Unit of measurement
        experimental_conditions: Experimental details
        split: Dataset split (train/val/test)
        quality_flags: Any quality concerns or notes
    """
    id: str
    dataset: str
    peptide: Peptide
    property_name: str
    property_value: float
    property_unit: str
    experimental_conditions: Optional[Dict[str, Any]] = None
    split: Optional[str] = None
    quality_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "dataset": self.dataset,
            "sequence": self.peptide.sequence,
            "modifications": [
                {
                    "type": m.modification_type.value,
                    "name": m.modification_name,
                    "position": m.position,
                    "notation": m.notation,
                    "details": m.details
                }
                for m in self.peptide.modifications
            ],
            "cyclization_info": self.peptide.cyclization_info,
            "original_notation": self.peptide.original_notation,
            "property_name": self.property_name,
            "property_value": self.property_value,
            "property_unit": self.property_unit,
            "experimental_conditions": self.experimental_conditions,
            "split": self.split,
            "quality_flags": self.quality_flags,
            "metadata": self.peptide.metadata
        }


@dataclass
class EditPair:
    """
    Pair of peptides representing a single edit.

    This is the core structure for anchor-aware edit modeling.

    Attributes:
        source: Source peptide
        target: Target peptide (after edit)
        edit_type: Type of edit (substitution/insertion/deletion/modification)
        anchor_position: Position of the edit in source sequence
        edit_description: Human-readable description
        property_change: Change in measured property (target - source)
    """
    source: DataPoint
    target: DataPoint
    edit_type: str
    anchor_position: int
    edit_description: str
    property_change: float

    def __post_init__(self):
        """Validate edit pair."""
        assert self.source.dataset == self.target.dataset, \
            "Edit pair must be from same dataset"
        assert self.source.property_name == self.target.property_name, \
            "Edit pair must measure same property"

    def get_context_window(self, window_size: int = 3) -> Tuple[str, str]:
        """
        Get sequence context around edit position.

        Args:
            window_size: Number of residues on each side

        Returns:
            (source_context, target_context)
        """
        start = max(0, self.anchor_position - window_size)
        end = min(
            len(self.source.peptide.sequence),
            self.anchor_position + window_size + 1
        )

        source_context = self.source.peptide.sequence[start:end]

        # Target might be different length due to indels
        end_target = min(
            len(self.target.peptide.sequence),
            self.anchor_position + window_size + 1
        )
        target_context = self.target.peptide.sequence[start:end_target]

        return source_context, target_context


def calculate_edit_distance(seq1: str, seq2: str) -> int:
    """
    Calculate Levenshtein edit distance between sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Edit distance
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    return dp[m][n]


def identify_edit_pairs(
    datapoints: List[DataPoint],
    max_edit_distance: int = 1,
    same_length_only: bool = False
) -> List[EditPair]:
    """
    Identify edit pairs from a list of datapoints.

    Args:
        datapoints: List of datapoints to search
        max_edit_distance: Maximum allowed edit distance
        same_length_only: Only consider same-length peptides

    Returns:
        List of identified edit pairs
    """
    # This is a placeholder - actual implementation depends on dataset structure
    # Will be refined during processing stage
    raise NotImplementedError(
        "Edit pair identification will be implemented based on actual dataset structure"
    )
