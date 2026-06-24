"""
Strict data auditing framework for PEM Stage 1 census.

This module implements conservative auditing with explicit classification
of anchor resolvability and no optimistic assumptions.

Anchor Classification Levels:
- EXPLICIT_ANCHOR: Direct annotation of edit position
- WEAKLY_INFERABLE: Can be inferred with reasonable confidence (flagged as inferred)
- NOT_RESOLVABLE: Cannot determine anchor position

Conservative Filtering Principles:
- No silent assumptions
- Separate different assay types strictly
- Log every exclusion with detailed reason
- Track ambiguous cases separately
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from collections import defaultdict, Counter
import pandas as pd
import re


class AnchorResolvability(Enum):
    """Classification of anchor position resolvability."""
    EXPLICIT_ANCHOR = "explicit_anchor"
    WEAKLY_INFERABLE = "weakly_inferable"
    NOT_RESOLVABLE = "not_resolvable"
    UNKNOWN = "unknown"


class ExclusionReason(Enum):
    """Standardized exclusion reasons for audit trail."""
    MISSING_SEQUENCE = "missing_sequence"
    UNPARSEABLE_SEQUENCE = "unparseable_sequence"
    MISSING_LABEL = "missing_label"
    NON_NUMERIC_LABEL = "non_numeric_label"
    MISSING_MODIFICATION = "missing_modification"
    AMBIGUOUS_MODIFICATION = "ambiguous_modification"
    NO_ANCHOR = "no_anchor_resolvable"
    WEAK_ANCHOR_ONLY = "weak_anchor_only"
    MULTI_EDIT = "multi_edit"
    DUPLICATE = "duplicate"
    WRONG_ASSAY_TYPE = "wrong_assay_type"
    MISSING_METADATA = "missing_metadata"
    INVALID_FORMAT = "invalid_format"
    OTHER = "other"


@dataclass
class CensusMetrics:
    """
    Comprehensive census metrics for a dataset.

    Tracks samples through progressive filtering stages with
    strict accounting of exclusion reasons.
    """
    dataset_name: str

    # Raw counts
    raw_sample_count: int = 0

    # Parseability
    parseable_sample_count: int = 0
    unparseable_samples: int = 0

    # Core fields
    sequence_available_count: int = 0
    missing_sequence_count: int = 0

    modification_available_count: int = 0
    no_modification_count: int = 0
    ambiguous_modification_count: int = 0

    # Anchor resolvability (strict classification)
    explicit_anchor_count: int = 0
    weakly_inferable_anchor_count: int = 0
    not_resolvable_anchor_count: int = 0

    # Label quality
    continuous_label_count: int = 0
    ordinal_label_count: int = 0
    missing_label_count: int = 0
    non_numeric_label_count: int = 0

    # Metadata
    assay_metadata_available_count: int = 0
    missing_metadata_count: int = 0

    # Data quality issues
    duplicate_count: int = 0
    multi_edit_count: int = 0
    single_edit_count: int = 0

    # Final usable counts (conservative)
    usable_with_explicit_anchor: int = 0
    usable_with_weak_anchor: int = 0
    usable_total: int = 0

    # Assay-specific tracking
    assay_type_distribution: Dict[str, int] = field(default_factory=dict)

    # Split feasibility notes
    split_feasibility_notes: List[str] = field(default_factory=list)

    # Detailed exclusion tracking
    exclusion_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SampleAuditRecord:
    """
    Individual sample audit record.

    Tracks detailed information for each sample including
    exclusion reasons and flags.
    """
    sample_id: str

    # Raw data
    raw_sequence: Optional[str] = None
    raw_modification: Optional[str] = None
    raw_label: Optional[Any] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    # Parsing results
    is_parseable: bool = False
    parse_errors: List[str] = field(default_factory=list)

    # Validation flags
    has_sequence: bool = False
    has_modification: bool = False
    has_label: bool = False
    has_metadata: bool = False

    # Anchor classification
    anchor_resolvability: AnchorResolvability = AnchorResolvability.UNKNOWN
    anchor_position: Optional[int] = None
    anchor_inference_method: Optional[str] = None

    # Edit classification
    is_single_edit: bool = False
    edit_count: Optional[int] = None

    # Quality flags
    is_duplicate: bool = False
    is_ambiguous: bool = False

    # Exclusion tracking
    is_excluded: bool = False
    exclusion_reasons: List[ExclusionReason] = field(default_factory=list)
    exclusion_notes: str = ""

    # Assay classification
    assay_type: Optional[str] = None

    def add_exclusion(self, reason: ExclusionReason, note: str = ""):
        """Add an exclusion reason with optional note."""
        self.is_excluded = True
        if reason not in self.exclusion_reasons:
            self.exclusion_reasons.append(reason)
        if note:
            if self.exclusion_notes:
                self.exclusion_notes += f"; {note}"
            else:
                self.exclusion_notes = note

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame export."""
        return {
            "sample_id": self.sample_id,
            "raw_sequence": self.raw_sequence,
            "raw_modification": self.raw_modification,
            "raw_label": self.raw_label,
            "is_parseable": self.is_parseable,
            "has_sequence": self.has_sequence,
            "has_modification": self.has_modification,
            "has_label": self.has_label,
            "anchor_resolvability": self.anchor_resolvability.value,
            "anchor_position": self.anchor_position,
            "anchor_inference_method": self.anchor_inference_method,
            "is_single_edit": self.is_single_edit,
            "edit_count": self.edit_count,
            "is_duplicate": self.is_duplicate,
            "is_excluded": self.is_excluded,
            "exclusion_reasons": ";".join([r.value for r in self.exclusion_reasons]),
            "exclusion_notes": self.exclusion_notes,
            "assay_type": self.assay_type
        }


class DatasetAuditor:
    """
    Strict dataset auditor with conservative filtering.

    Implements progressive filtering with detailed tracking
    of exclusion reasons and audit records.
    """

    def __init__(self, dataset_name: str, strict_mode: bool = True):
        """
        Initialize auditor.

        Args:
            dataset_name: Name of dataset being audited
            strict_mode: If True, apply most conservative filters
        """
        self.dataset_name = dataset_name
        self.strict_mode = strict_mode

        self.metrics = CensusMetrics(dataset_name=dataset_name)
        self.audit_records: List[SampleAuditRecord] = []
        self.exclusion_examples: Dict[str, List[Dict]] = defaultdict(list)

    def audit_sample(
        self,
        sample_id: str,
        raw_data: Dict[str, Any],
        sequence_field: str = "sequence",
        modification_field: Optional[str] = "modification",
        label_field: str = "label",
        assay_field: Optional[str] = "assay_type"
    ) -> SampleAuditRecord:
        """
        Audit a single sample with strict validation.

        Args:
            sample_id: Unique sample identifier
            raw_data: Raw data dictionary
            sequence_field: Column name for sequence
            modification_field: Column name for modification info
            label_field: Column name for label/target
            assay_field: Column name for assay type

        Returns:
            SampleAuditRecord with complete audit information
        """
        record = SampleAuditRecord(sample_id=sample_id)

        # Extract raw data
        record.raw_sequence = raw_data.get(sequence_field)
        record.raw_modification = raw_data.get(modification_field) if modification_field else None
        record.raw_label = raw_data.get(label_field)
        record.assay_type = raw_data.get(assay_field) if assay_field else None

        # Validation: Sequence
        if pd.isna(record.raw_sequence) or not record.raw_sequence:
            record.add_exclusion(
                ExclusionReason.MISSING_SEQUENCE,
                "No sequence provided"
            )
        else:
            # Try to parse sequence
            if self._validate_sequence(record.raw_sequence):
                record.has_sequence = True
                record.is_parseable = True
            else:
                record.add_exclusion(
                    ExclusionReason.UNPARSEABLE_SEQUENCE,
                    f"Cannot parse sequence: {record.raw_sequence[:50]}"
                )

        # Validation: Modification
        if modification_field:
            if pd.isna(record.raw_modification) or not record.raw_modification:
                record.add_exclusion(
                    ExclusionReason.MISSING_MODIFICATION,
                    "No modification information"
                )
            else:
                # Check for ambiguous modification annotations
                if self._is_ambiguous_modification(record.raw_modification):
                    record.is_ambiguous = True
                    record.add_exclusion(
                        ExclusionReason.AMBIGUOUS_MODIFICATION,
                        f"Ambiguous modification: {str(record.raw_modification)[:50]}"
                    )
                else:
                    record.has_modification = True

        # Validation: Label
        if pd.isna(record.raw_label):
            record.add_exclusion(
                ExclusionReason.MISSING_LABEL,
                "No label/target value"
            )
        else:
            try:
                float(record.raw_label)
                record.has_label = True
            except (ValueError, TypeError):
                record.add_exclusion(
                    ExclusionReason.NON_NUMERIC_LABEL,
                    f"Non-numeric label: {record.raw_label}"
                )

        # Classify anchor resolvability (if sequence and modification present)
        if record.has_sequence and record.has_modification:
            record.anchor_resolvability = self._classify_anchor_resolvability(
                record.raw_sequence,
                record.raw_modification
            )

            if record.anchor_resolvability == AnchorResolvability.NOT_RESOLVABLE:
                record.add_exclusion(
                    ExclusionReason.NO_ANCHOR,
                    "Cannot resolve anchor position"
                )
            elif record.anchor_resolvability == AnchorResolvability.WEAKLY_INFERABLE:
                # Flag but don't necessarily exclude
                record.anchor_inference_method = "heuristic"
                if self.strict_mode:
                    record.add_exclusion(
                        ExclusionReason.WEAK_ANCHOR_ONLY,
                        "Only weak anchor inference available (strict mode)"
                    )

        return record

    def _validate_sequence(self, sequence: str) -> bool:
        """
        Validate sequence format.

        Args:
            sequence: Raw sequence string

        Returns:
            True if sequence appears parseable
        """
        if not isinstance(sequence, str):
            return False

        # Remove common decorators
        seq_clean = str(sequence).strip().upper()

        # Check if contains reasonable amino acid characters
        # (Allow some flexibility for modification notation)
        valid_chars = set('ACDEFGHIKLMNPQRSTVWY[]()-_.*')

        if not seq_clean:
            return False

        # At least some standard amino acids should be present
        aa_chars = set('ACDEFGHIKLMNPQRSTVWY')
        has_aa = any(c in aa_chars for c in seq_clean)

        return has_aa

    def _is_ambiguous_modification(self, modification: str) -> bool:
        """
        Check if modification annotation is ambiguous.

        Args:
            modification: Modification string

        Returns:
            True if ambiguous
        """
        if not isinstance(modification, str):
            return True

        mod_lower = modification.lower()

        # Ambiguous patterns
        ambiguous_patterns = [
            "unknown",
            "unclear",
            "multiple",
            "various",
            "unspecified",
            "?",
            "tbd",
            "see notes"
        ]

        return any(pattern in mod_lower for pattern in ambiguous_patterns)

    def _classify_anchor_resolvability(
        self,
        sequence: str,
        modification: str
    ) -> AnchorResolvability:
        """
        Classify anchor resolvability level.

        This is a placeholder - actual implementation will depend on
        dataset-specific format discovered during census.

        Args:
            sequence: Sequence string
            modification: Modification string

        Returns:
            AnchorResolvability classification
        """
        # This will be implemented based on actual data formats
        # For now, mark as unknown
        return AnchorResolvability.UNKNOWN

    def compute_metrics(self) -> CensusMetrics:
        """
        Compute final census metrics from audit records.

        Returns:
            Populated CensusMetrics object
        """
        self.metrics.raw_sample_count = len(self.audit_records)

        for record in self.audit_records:
            # Parseability
            if record.is_parseable:
                self.metrics.parseable_sample_count += 1
            else:
                self.metrics.unparseable_samples += 1

            # Core fields
            if record.has_sequence:
                self.metrics.sequence_available_count += 1
            else:
                self.metrics.missing_sequence_count += 1

            if record.has_modification:
                self.metrics.modification_available_count += 1
            elif ExclusionReason.MISSING_MODIFICATION in record.exclusion_reasons:
                self.metrics.no_modification_count += 1
            elif ExclusionReason.AMBIGUOUS_MODIFICATION in record.exclusion_reasons:
                self.metrics.ambiguous_modification_count += 1

            # Anchor resolvability
            if record.anchor_resolvability == AnchorResolvability.EXPLICIT_ANCHOR:
                self.metrics.explicit_anchor_count += 1
            elif record.anchor_resolvability == AnchorResolvability.WEAKLY_INFERABLE:
                self.metrics.weakly_inferable_anchor_count += 1
            elif record.anchor_resolvability == AnchorResolvability.NOT_RESOLVABLE:
                self.metrics.not_resolvable_anchor_count += 1

            # Labels
            if record.has_label:
                # Assume continuous for now (can refine later)
                self.metrics.continuous_label_count += 1
            elif ExclusionReason.MISSING_LABEL in record.exclusion_reasons:
                self.metrics.missing_label_count += 1
            elif ExclusionReason.NON_NUMERIC_LABEL in record.exclusion_reasons:
                self.metrics.non_numeric_label_count += 1

            # Duplicates and edits
            if record.is_duplicate:
                self.metrics.duplicate_count += 1
            if record.is_single_edit:
                self.metrics.single_edit_count += 1
            if record.edit_count and record.edit_count > 1:
                self.metrics.multi_edit_count += 1

            # Assay types
            if record.assay_type:
                self.metrics.assay_type_distribution[record.assay_type] = \
                    self.metrics.assay_type_distribution.get(record.assay_type, 0) + 1

            # Exclusions
            for reason in record.exclusion_reasons:
                self.metrics.exclusion_counts[reason.value] += 1

            # Usable counts (conservative)
            if not record.is_excluded:
                self.metrics.usable_total += 1

                if record.anchor_resolvability == AnchorResolvability.EXPLICIT_ANCHOR:
                    self.metrics.usable_with_explicit_anchor += 1
                elif record.anchor_resolvability == AnchorResolvability.WEAKLY_INFERABLE:
                    self.metrics.usable_with_weak_anchor += 1

        # Split feasibility assessment
        self._assess_split_feasibility()

        return self.metrics

    def _assess_split_feasibility(self):
        """Assess whether dataset is large enough for train/val/test split."""
        min_viable_size = 100
        recommended_size = 300

        usable = self.metrics.usable_total

        if usable < min_viable_size:
            self.metrics.split_feasibility_notes.append(
                f"INSUFFICIENT: Only {usable} usable samples (minimum {min_viable_size} recommended)"
            )
        elif usable < recommended_size:
            self.metrics.split_feasibility_notes.append(
                f"MARGINAL: {usable} usable samples (recommended {recommended_size}+ for robust evaluation)"
            )
        else:
            self.metrics.split_feasibility_notes.append(
                f"SUFFICIENT: {usable} usable samples for train/val/test split"
            )

        # Check class balance for edit types if available
        if self.metrics.single_edit_count > 0:
            edit_ratio = self.metrics.single_edit_count / max(usable, 1)
            if edit_ratio < 0.1:
                self.metrics.split_feasibility_notes.append(
                    f"WARNING: Low single-edit ratio ({edit_ratio:.2%})"
                )

    def save_exclusion_csv(self, output_path: str):
        """
        Save detailed exclusion records to CSV.

        Args:
            output_path: Path to save exclusion CSV
        """
        excluded_records = [
            record.to_dict()
            for record in self.audit_records
            if record.is_excluded
        ]

        if excluded_records:
            df = pd.DataFrame(excluded_records)
            df.to_csv(output_path, index=False)
            return len(excluded_records)
        return 0

    def save_all_records_csv(self, output_path: str):
        """
        Save all audit records to CSV.

        Args:
            output_path: Path to save CSV
        """
        all_records = [record.to_dict() for record in self.audit_records]

        if all_records:
            df = pd.DataFrame(all_records)
            df.to_csv(output_path, index=False)
            return len(all_records)
        return 0
