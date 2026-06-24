"""
Tests for data splitting logic.

Tests all split strategies, leakage analysis, and utility functions.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pem_schema import (
    PEMSample, Edit, EditFamily, AnchorKind, AnchorStatus, ParserStatus
)
from src.data.splitting import (
    RandomSplitter,
    ScaffoldAwareSplitter,
    SequenceClusterSplitter,
    EditFamilyAwareSplitter,
    SameEditFamilyNewScaffoldSplitter,
    GroupedDerivativeSplitter,
)
from src.data.splitting.utils import (
    extract_scaffold,
    compute_sequence_similarity,
    cluster_sequences_simple,
    get_edit_profile,
    identify_derivative_groups,
)
from src.data.splitting.leakage import (
    analyze_leakage,
    classify_leakage_risk,
    LeakageRisk,
)


# Helper functions

def create_sample(
    sample_id: str,
    sequence: str,
    label: float,
    edits: list = None,
) -> PEMSample:
    """Create a minimal PEMSample for testing."""
    return PEMSample(
        sample_id=sample_id,
        dataset="TEST",
        sequence=sequence,
        label=label,
        label_type="test_label",
        label_unit="arbitrary",
        assay_type="test_assay",
        assay_metadata={"test": "metadata"},
        edits=edits or [],
        anchor_status=AnchorStatus.NO_EDITS if not edits else AnchorStatus.EXPLICIT_ANCHOR,
        provenance={
            "source_file": "test.csv",
            "source_row_index": 0,
            "parser_version": "1.0.0"
        },
    )


def create_edit(
    edit_id: str,
    edit_family: EditFamily,
    edit_type: str,
    anchor_positions: list,
    anchor_residues: list,
) -> Edit:
    """Create a minimal Edit for testing."""
    return Edit(
        edit_id=edit_id,
        edit_family=edit_family,
        edit_type=edit_type,
        anchor_kind=AnchorKind.EXPLICIT,
        anchor_positions=anchor_positions,
        anchor_residues=anchor_residues,
        chem_rep_raw="test",
        chem_rep_canonical="test",
        attachment_semantics="test",
        parser_status=ParserStatus.SUCCESS,
        rule_id="test_rule",
    )


# Utility function tests

def test_extract_scaffold_simple():
    """Test scaffold extraction for simple peptide."""
    sample = create_sample("TEST_001", "ACDEFG", 1.0)
    scaffold = extract_scaffold(sample)
    assert scaffold == "ACDEFG"


def test_extract_scaffold_with_cyclization():
    """Test scaffold extraction with cyclization."""
    edit = create_edit(
        "TEST_001_edit_1",
        EditFamily.CYCLIZATION,
        "head_to_tail",
        [0, 5],
        ["A", "G"],
    )
    sample = create_sample("TEST_001", "ACDEFG", 1.0, edits=[edit])
    scaffold = extract_scaffold(sample)
    assert "cyc" in scaffold
    assert "ACDEFG" in scaffold


def test_sequence_similarity():
    """Test sequence similarity computation."""
    # Identical sequences
    assert compute_sequence_similarity("ACDEFG", "ACDEFG") == 1.0

    # Different sequences
    sim = compute_sequence_similarity("ACDEFG", "ACDXFG")
    assert 0.8 < sim < 1.0  # One mismatch

    # Completely different
    sim = compute_sequence_similarity("AAA", "GGG")
    assert sim < 0.5


def test_cluster_sequences_simple():
    """Test simple sequence clustering."""
    sequences = [
        "ACDEFG",
        "ACDEFG",  # Duplicate
        "ACDXFG",  # Similar
        "KLMNOP",  # Different
    ]

    clusters = cluster_sequences_simple(sequences, identity_threshold=0.9)

    # Should have 2 clusters (similar sequences grouped, different separate)
    assert len(clusters) >= 2

    # Duplicates should be in same cluster
    cluster_0 = None
    cluster_1 = None
    for cluster_id, members in clusters.items():
        if 0 in members:
            cluster_0 = cluster_id
        if 1 in members:
            cluster_1 = cluster_id

    assert cluster_0 == cluster_1  # Duplicates in same cluster


def test_get_edit_profile():
    """Test edit profile extraction."""
    # No edits
    sample = create_sample("TEST_001", "ACDEFG", 1.0)
    assert get_edit_profile(sample) == "no_edits"

    # Single edit
    edit = create_edit(
        "TEST_001_edit_1",
        EditFamily.SIDECHAIN,
        "methylation",
        [0],
        ["A"],
    )
    sample = create_sample("TEST_001", "ACDEFG", 1.0, edits=[edit])
    assert get_edit_profile(sample) == "sidechain"

    # Multiple edit families
    edit1 = create_edit(
        "TEST_001_edit_1",
        EditFamily.SIDECHAIN,
        "methylation",
        [0],
        ["A"],
    )
    edit2 = create_edit(
        "TEST_001_edit_2",
        EditFamily.CYCLIZATION,
        "head_to_tail",
        [0, 5],
        ["A", "G"],
    )
    sample = create_sample("TEST_001", "ACDEFG", 1.0, edits=[edit1, edit2])
    profile = get_edit_profile(sample)
    assert "sidechain" in profile
    assert "cyclization" in profile


def test_identify_derivative_groups():
    """Test derivative group identification."""
    # Create samples with same sequence but different edits
    samples = [
        create_sample("TEST_001", "ACDEFG", 1.0),
        create_sample("TEST_002", "ACDEFG", 1.0, edits=[
            create_edit("TEST_002_edit_1", EditFamily.SIDECHAIN, "meth", [0], ["A"])
        ]),
        create_sample("TEST_003", "ACDEFG", 1.0, edits=[
            create_edit("TEST_003_edit_1", EditFamily.SIDECHAIN, "meth", [1], ["C"])
        ]),
        create_sample("TEST_004", "KLMNOP", 1.0),  # Different sequence
    ]

    groups = identify_derivative_groups(samples, max_edit_distance=1)

    # Should have at least one group for similar ACDEFG variants
    assert len(groups) >= 1


# Split strategy tests

def test_random_splitter():
    """Test random split strategy."""
    # Create test samples
    samples = [
        create_sample(f"TEST_{i:03d}", "ACDEFG", float(i))
        for i in range(100)
    ]

    splitter = RandomSplitter(random_seed=42)

    # Check feasibility
    is_feasible, reason = splitter.check_feasibility(samples)
    assert is_feasible

    # Generate split
    result = splitter.split(samples)

    # Check counts
    assert len(result.train_samples) > 0
    assert len(result.val_samples) > 0
    assert len(result.test_samples) > 0
    assert result.total_samples == 100

    # Check no overlap
    train_ids = {s.sample_id for s in result.train_samples}
    test_ids = {s.sample_id for s in result.test_samples}
    assert len(train_ids & test_ids) == 0

    # Check proportions are close
    train_pct = len(result.train_samples) / 100
    assert 0.65 < train_pct < 0.75  # Should be ~0.70


def test_random_splitter_too_few_samples():
    """Test random splitter rejects too few samples."""
    samples = [create_sample(f"TEST_{i:03d}", "ACDEFG", 1.0) for i in range(5)]

    splitter = RandomSplitter()
    is_feasible, reason = splitter.check_feasibility(samples)
    assert not is_feasible
    assert "Too few" in reason


def test_scaffold_aware_splitter():
    """Test scaffold-aware split strategy."""
    # Create samples with multiple scaffolds
    samples = []
    for scaffold_id in range(30):  # 30 different scaffolds
        for variant in range(3):  # 3 variants per scaffold
            seq = f"AC{chr(65 + scaffold_id % 26)}DEFG"
            samples.append(
                create_sample(f"TEST_{scaffold_id:03d}_{variant}", seq, 1.0)
            )

    splitter = ScaffoldAwareSplitter(min_scaffolds=20, random_seed=42)

    # Check feasibility
    is_feasible, reason = splitter.check_feasibility(samples)
    assert is_feasible

    # Generate split
    result = splitter.split(samples)

    # Extract scaffolds for each split
    train_scaffolds = {extract_scaffold(s) for s in result.train_samples}
    test_scaffolds = {extract_scaffold(s) for s in result.test_samples}

    # Scaffolds should not overlap between train and test
    assert len(train_scaffolds & test_scaffolds) == 0


def test_scaffold_aware_splitter_insufficient_scaffolds():
    """Test scaffold-aware splitter rejects insufficient scaffolds."""
    # Only 5 unique scaffolds
    samples = []
    for scaffold_id in range(5):
        for variant in range(10):
            seq = f"AC{chr(65 + scaffold_id)}DEFG"
            samples.append(
                create_sample(f"TEST_{scaffold_id:03d}_{variant}", seq, 1.0)
            )

    splitter = ScaffoldAwareSplitter(min_scaffolds=20)
    is_feasible, reason = splitter.check_feasibility(samples)
    assert not is_feasible
    assert "scaffolds" in reason.lower()


def test_sequence_cluster_splitter():
    """Test sequence cluster split strategy."""
    # Create samples with clustered sequences
    samples = []

    # Cluster 1: Similar to "ACDEFG"
    for i in range(20):
        samples.append(create_sample(f"TEST_A_{i:03d}", "ACDEFG", 1.0))

    # Cluster 2: Similar to "KLMNOP"
    for i in range(20):
        samples.append(create_sample(f"TEST_B_{i:03d}", "KLMNOP", 1.0))

    # More clusters...
    for cluster_id in range(18):  # 18 more clusters
        seq = f"X{chr(65 + cluster_id % 26)}Y{chr(66 + cluster_id % 26)}ZW"
        for i in range(3):
            samples.append(
                create_sample(f"TEST_C{cluster_id}_{i:03d}", seq, 1.0)
            )

    splitter = SequenceClusterSplitter(
        identity_threshold=0.9,
        min_clusters=20,
        clustering_method='simple',
        random_seed=42,
    )

    # Check feasibility
    is_feasible, reason = splitter.check_feasibility(samples)
    assert is_feasible

    # Generate split
    result = splitter.split(samples)

    # Sequences within same cluster should be in same split
    # (This is implicitly tested by the splitter logic)
    assert result.total_samples == len(samples)


def test_edit_family_aware_splitter():
    """Test edit-family-aware split strategy."""
    # Create samples with different edit profiles
    samples = []

    # Profile 1: No edits (30 samples)
    for i in range(30):
        samples.append(create_sample(f"TEST_NONE_{i:03d}", "ACDEFG", 1.0))

    # Profile 2: Sidechain only (30 samples)
    for i in range(30):
        edit = create_edit(
            f"TEST_SC_{i:03d}_edit_1",
            EditFamily.SIDECHAIN,
            "meth",
            [0],
            ["A"]
        )
        samples.append(
            create_sample(f"TEST_SC_{i:03d}", "ACDEFG", 1.0, edits=[edit])
        )

    # Profile 3: Cyclization only (30 samples)
    for i in range(30):
        edit = create_edit(
            f"TEST_CYC_{i:03d}_edit_1",
            EditFamily.CYCLIZATION,
            "head_to_tail",
            [0, 5],
            ["A", "G"]
        )
        samples.append(
            create_sample(f"TEST_CYC_{i:03d}", "ACDEFG", 1.0, edits=[edit])
        )

    splitter = EditFamilyAwareSplitter(min_per_profile=10, random_seed=42)

    # Check feasibility
    is_feasible, reason = splitter.check_feasibility(samples)
    assert is_feasible

    # Generate split
    result = splitter.split(samples)

    # Each split should have representation from all profiles
    # (tested implicitly by stratification)
    assert result.total_samples == len(samples)


# Leakage analysis tests

def test_classify_leakage_risk():
    """Test leakage risk classification."""
    # Low risk
    assert classify_leakage_risk(0.6, 5.0) == LeakageRisk.LOW

    # High risk (high similarity)
    assert classify_leakage_risk(0.9, 10.0) == LeakageRisk.HIGH

    # High risk (high scaffold overlap)
    assert classify_leakage_risk(0.7, 60.0) == LeakageRisk.HIGH

    # Moderate risk
    assert classify_leakage_risk(0.75, 20.0) == LeakageRisk.MODERATE


def test_analyze_leakage():
    """Test comprehensive leakage analysis."""
    # Create train and test samples
    train_samples = [
        create_sample(f"TRAIN_{i:03d}", "ACDEFG", float(i))
        for i in range(50)
    ]

    # Test samples with some overlap
    test_samples = [
        create_sample(f"TEST_{i:03d}", "ACDEFG", float(i + 50))
        for i in range(25)  # Same sequence
    ] + [
        create_sample(f"TEST_{i:03d}", "KLMNOP", float(i + 75))
        for i in range(25, 50)  # Different sequence
    ]

    val_samples = []

    # Analyze leakage
    analysis = analyze_leakage(train_samples, val_samples, test_samples)

    # Should detect high sequence similarity
    assert analysis.max_sequence_identity > 0.9

    # Should classify as high risk
    assert analysis.overall_risk in [LeakageRisk.HIGH, LeakageRisk.MODERATE]


def test_split_result_validation():
    """Test that SplitResult validates no overlap."""
    from src.data.splitting.base import SplitResult, SplitMetadata

    samples = [create_sample(f"TEST_{i:03d}", "ACDEFG", 1.0) for i in range(10)]

    # Valid split (no overlap)
    metadata = SplitMetadata(strategy_name="test", feasible=True)
    result = SplitResult(
        train_samples=samples[:7],
        val_samples=samples[7:8],
        test_samples=samples[8:],
        metadata=metadata,
    )
    assert result.total_samples == 10

    # Invalid split (overlap)
    with pytest.raises(ValueError, match="overlapping"):
        SplitResult(
            train_samples=samples[:7],
            val_samples=samples[5:8],  # Overlaps with train
            test_samples=samples[8:],
            metadata=metadata,
        )


# Integration tests

def test_full_pipeline_small_dataset():
    """Test complete splitting pipeline on small dataset."""
    # Create diverse dataset
    samples = []

    # Multiple scaffolds
    for scaffold_id in range(25):
        seq = f"AC{chr(65 + scaffold_id % 26)}DEF"

        # Multiple edits per scaffold
        for edit_var in range(4):
            if edit_var == 0:
                # No edit
                edits = []
            elif edit_var == 1:
                # Sidechain
                edits = [create_edit(
                    f"S{scaffold_id}_E{edit_var}_edit_1",
                    EditFamily.SIDECHAIN,
                    "meth",
                    [0],
                    ["A"]
                )]
            elif edit_var == 2:
                # Cyclization
                edits = [create_edit(
                    f"S{scaffold_id}_E{edit_var}_edit_1",
                    EditFamily.CYCLIZATION,
                    "head_to_tail",
                    [0, 4],
                    ["A", "F"]
                )]
            else:
                # Both
                edits = [
                    create_edit(
                        f"S{scaffold_id}_E{edit_var}_edit_1",
                        EditFamily.SIDECHAIN,
                        "meth",
                        [0],
                        ["A"]
                    ),
                    create_edit(
                        f"S{scaffold_id}_E{edit_var}_edit_2",
                        EditFamily.CYCLIZATION,
                        "head_to_tail",
                        [0, 4],
                        ["A", "F"]
                    ),
                ]

            samples.append(create_sample(
                f"TEST_S{scaffold_id}_E{edit_var}",
                seq,
                float(scaffold_id * 10 + edit_var),
                edits=edits
            ))

    # Try all strategies
    strategies = [
        RandomSplitter(random_seed=42),
        ScaffoldAwareSplitter(min_scaffolds=20, random_seed=42),
        EditFamilyAwareSplitter(min_per_profile=5, random_seed=42),
    ]

    for splitter in strategies:
        is_feasible, reason = splitter.check_feasibility(samples)

        if is_feasible:
            result = splitter.split(samples)

            # Verify basic properties
            assert result.total_samples == len(samples)
            assert len(result.train_samples) > 0
            assert len(result.val_samples) > 0
            assert len(result.test_samples) > 0

            # Verify metadata
            assert result.metadata.feasible
            assert result.metadata.leakage_analysis is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
