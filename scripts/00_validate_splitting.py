#!/usr/bin/env python3
"""
Validate splitting implementation with synthetic data.

Creates synthetic PEM samples and demonstrates all split strategies.
"""

import sys
from pathlib import Path

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


def create_synthetic_sample(
    sample_id: str,
    sequence: str,
    label: float,
    edits: list = None,
) -> PEMSample:
    """Create a synthetic PEMSample."""
    return PEMSample(
        sample_id=sample_id,
        dataset="SYNTHETIC",
        sequence=sequence,
        label=label,
        label_type="synthetic_property",
        label_unit="arbitrary",
        assay_type="synthetic_assay",
        assay_metadata={"synthetic": True},
        edits=edits or [],
        anchor_status=AnchorStatus.NO_EDITS if not edits else AnchorStatus.EXPLICIT_ANCHOR,
        provenance={
            "source_file": "synthetic.py",
            "source_row_index": 0,
            "parser_version": "1.0.0"
        },
    )


def create_synthetic_edit(
    edit_id: str,
    edit_family: EditFamily,
    positions: list,
    residues: list,
) -> Edit:
    """Create a synthetic Edit."""
    return Edit(
        edit_id=edit_id,
        edit_family=edit_family,
        edit_type=f"{edit_family}_modification",
        anchor_kind=AnchorKind.EXPLICIT,
        anchor_positions=positions,
        anchor_residues=residues,
        chem_rep_raw="synthetic",
        chem_rep_canonical="synthetic",
        attachment_semantics="synthetic",
        parser_status=ParserStatus.SUCCESS,
        rule_id="synthetic_rule",
    )


def generate_synthetic_dataset(n_scaffolds=30, samples_per_scaffold=4):
    """Generate synthetic dataset with diverse properties."""
    samples = []
    sample_idx = 0

    for scaffold_id in range(n_scaffolds):
        # Generate scaffold sequence
        base_seq = f"AC{chr(65 + scaffold_id % 26)}DEFG"

        for variant in range(samples_per_scaffold):
            edits = []

            # Variant 0: No edits
            # Variant 1: Sidechain modification
            if variant == 1:
                edits = [create_synthetic_edit(
                    f"SYNTH_{sample_idx:04d}_edit_1",
                    EditFamily.SIDECHAIN,
                    [0],
                    [base_seq[0]]
                )]

            # Variant 2: Cyclization
            elif variant == 2:
                edits = [create_synthetic_edit(
                    f"SYNTH_{sample_idx:04d}_edit_1",
                    EditFamily.CYCLIZATION,
                    [0, len(base_seq) - 1],
                    [base_seq[0], base_seq[-1]]
                )]

            # Variant 3: Both
            elif variant == 3:
                edits = [
                    create_synthetic_edit(
                        f"SYNTH_{sample_idx:04d}_edit_1",
                        EditFamily.SIDECHAIN,
                        [0],
                        [base_seq[0]]
                    ),
                    create_synthetic_edit(
                        f"SYNTH_{sample_idx:04d}_edit_2",
                        EditFamily.CYCLIZATION,
                        [0, len(base_seq) - 1],
                        [base_seq[0], base_seq[-1]]
                    ),
                ]

            sample = create_synthetic_sample(
                f"SYNTH_{sample_idx:04d}",
                base_seq,
                float(scaffold_id * 10 + variant),
                edits=edits
            )

            samples.append(sample)
            sample_idx += 1

    return samples


def test_strategy(splitter, samples, strategy_name):
    """Test a split strategy."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {strategy_name}")
    print(f"{'=' * 60}")

    # Check feasibility
    is_feasible, reason = splitter.check_feasibility(samples)

    if not is_feasible:
        print(f"❌ NOT FEASIBLE: {reason}")
        return False

    print(f"✓ Feasibility check passed")

    # Generate split
    try:
        result = splitter.split(samples)
        print(f"✓ Split generated successfully")

        # Print statistics
        print(f"\nSplit Statistics:")
        print(f"  Train: {len(result.train_samples)} ({len(result.train_samples)/len(samples)*100:.1f}%)")
        print(f"  Val:   {len(result.val_samples)} ({len(result.val_samples)/len(samples)*100:.1f}%)")
        print(f"  Test:  {len(result.test_samples)} ({len(result.test_samples)/len(samples)*100:.1f}%)")
        print(f"  Total: {result.total_samples}")

        # Leakage analysis
        leakage = result.metadata.leakage_analysis
        print(f"\nLeakage Analysis:")
        print(f"  Max sequence identity: {leakage.max_sequence_identity:.3f}")
        print(f"  % test high similarity: {leakage.pct_test_high_similarity:.1f}%")
        print(f"  Shared scaffolds: {leakage.n_shared_scaffolds}")
        print(f"  % test shared scaffold: {leakage.pct_test_shared_scaffold:.1f}%")
        print(f"  Overall risk: {leakage.overall_risk.upper()}")

        # Strategy-specific info
        metadata = result.metadata
        if metadata.n_scaffolds:
            print(f"\nStrategy Details:")
            print(f"  Unique scaffolds: {metadata.n_scaffolds}")
        if metadata.n_clusters:
            print(f"  Sequence clusters: {metadata.n_clusters}")
        if metadata.n_edit_profiles:
            print(f"  Edit profiles: {metadata.n_edit_profiles}")
        if metadata.n_derivative_groups:
            print(f"  Derivative groups: {metadata.n_derivative_groups}")

        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main validation."""
    print("=" * 60)
    print("PEM Splitting Implementation Validation")
    print("=" * 60)

    # Generate synthetic dataset
    print("\nGenerating synthetic dataset...")
    samples = generate_synthetic_dataset(n_scaffolds=30, samples_per_scaffold=4)
    print(f"✓ Generated {len(samples)} synthetic samples")
    print(f"  - {len(set(s.sequence for s in samples))} unique scaffolds")
    print(f"  - {len(set(s.label for s in samples))} unique labels")

    # Test all strategies
    strategies = [
        (RandomSplitter(random_seed=42), "Random Split"),
        (ScaffoldAwareSplitter(min_scaffolds=20, random_seed=42), "Scaffold-Aware Split"),
        (SequenceClusterSplitter(
            identity_threshold=0.7,
            min_clusters=20,
            clustering_method='simple',
            random_seed=42
        ), "Sequence Cluster Split"),
        (EditFamilyAwareSplitter(min_per_profile=5, random_seed=42), "Edit-Family-Aware Split"),
        (SameEditFamilyNewScaffoldSplitter(
            min_scaffolds_per_family=3,
            random_seed=42
        ), "Same-Edit-Family / New-Scaffold Split"),
        (GroupedDerivativeSplitter(
            max_edit_distance=1,
            min_coverage=0.05,
            max_group_size=0.3,
            random_seed=42
        ), "Grouped Derivative Split"),
    ]

    results = {}
    for splitter, name in strategies:
        success = test_strategy(splitter, samples, name)
        results[name] = "✓ PASSED" if success else "✗ FAILED"

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, status in results.items():
        print(f"{name}: {status}")

    # Overall
    n_passed = sum(1 for status in results.values() if "PASSED" in status)
    n_total = len(results)
    print(f"\nOverall: {n_passed}/{n_total} strategies working")

    if n_passed == n_total:
        print("\n🎉 All strategies validated successfully!")
        return 0
    else:
        print(f"\n⚠️  {n_total - n_passed} strategies failed")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
