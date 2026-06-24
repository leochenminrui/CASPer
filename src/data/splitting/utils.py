"""
Utility functions for data splitting.

Includes sequence clustering, scaffold extraction, and derivative grouping.
"""

from typing import List, Dict, Set, Tuple, Optional
import subprocess
import tempfile
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from Bio import pairwise2
from Bio.Seq import Seq

from ..pem_schema import PEMSample, EditFamily


def extract_scaffold(sample: PEMSample) -> str:
    """
    Extract backbone scaffold from a sample.

    For peptides, scaffold is:
    - Canonical sequence (no modifications)
    - Cyclization topology (if applicable)

    Args:
        sample: PEMSample object

    Returns:
        Scaffold identifier string
    """
    # Base scaffold is just the sequence
    scaffold = sample.sequence

    # Check for cyclization edits
    cyclization_edits = [
        edit for edit in sample.edits
        if edit.edit_family == EditFamily.CYCLIZATION
    ]

    if cyclization_edits:
        # Include cyclization topology in scaffold
        # Sort positions to ensure consistent representation
        all_positions = []
        for edit in cyclization_edits:
            all_positions.extend(edit.anchor_positions)
        all_positions = sorted(set(all_positions))

        # Create scaffold with cyclization annotation
        scaffold = f"{scaffold}_cyc_{'_'.join(map(str, all_positions))}"

    return scaffold


def compute_sequence_similarity(seq1: str, seq2: str) -> float:
    """
    Compute sequence identity between two sequences.

    Uses global alignment (Needleman-Wunsch).

    Args:
        seq1: First sequence
        seq2: First sequence

    Returns:
        Sequence identity (0.0 to 1.0)
    """
    if seq1 == seq2:
        return 1.0

    # Use Bio.pairwise2 for alignment
    alignments = pairwise2.align.globalxx(seq1, seq2)

    if not alignments:
        return 0.0

    # Get best alignment
    best_alignment = alignments[0]
    aligned_seq1 = best_alignment.seqA
    aligned_seq2 = best_alignment.seqB

    # Compute identity
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b)
    identity = matches / max(len(seq1), len(seq2))

    return identity


def cluster_sequences_mmseqs2(
    sequences: List[str],
    identity_threshold: float = 0.7,
    coverage_threshold: float = 0.8,
) -> Dict[int, List[int]]:
    """
    Cluster sequences using MMseqs2.

    Args:
        sequences: List of sequences to cluster
        identity_threshold: Minimum sequence identity (default 0.7)
        coverage_threshold: Minimum coverage (default 0.8)

    Returns:
        Dictionary mapping cluster_id -> [sequence_indices]
    """
    # Check if mmseqs is available
    try:
        subprocess.run(
            ['mmseqs', 'version'],
            capture_output=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "MMseqs2 not found. Please install: "
            "conda install -c bioconda mmseqs2"
        )

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write sequences to FASTA
        fasta_file = tmpdir_path / "sequences.fasta"
        with open(fasta_file, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq_{i}\n{seq}\n")

        # Run MMseqs2
        db_file = tmpdir_path / "seqDB"
        cluster_db = tmpdir_path / "clusterDB"
        tmp_folder = tmpdir_path / "tmp"
        tmp_folder.mkdir()

        # Create sequence database
        subprocess.run(
            ['mmseqs', 'createdb', str(fasta_file), str(db_file)],
            capture_output=True,
            check=True
        )

        # Cluster
        subprocess.run([
            'mmseqs', 'cluster',
            str(db_file), str(cluster_db), str(tmp_folder),
            '--min-seq-id', str(identity_threshold),
            '-c', str(coverage_threshold),
            '--cov-mode', '0',  # Bidirectional coverage
        ], capture_output=True, check=True)

        # Create TSV output
        tsv_file = tmpdir_path / "clusters.tsv"
        subprocess.run([
            'mmseqs', 'createtsv',
            str(db_file), str(db_file), str(cluster_db), str(tsv_file)
        ], capture_output=True, check=True)

        # Parse clusters
        clusters = defaultdict(list)
        with open(tsv_file, 'r') as f:
            for line in f:
                rep, member = line.strip().split('\t')
                rep_id = int(rep.split('_')[1])
                member_id = int(member.split('_')[1])
                clusters[rep_id].append(member_id)

    return dict(clusters)


def cluster_sequences_cdhit(
    sequences: List[str],
    identity_threshold: float = 0.7,
) -> Dict[int, List[int]]:
    """
    Cluster sequences using CD-HIT.

    Args:
        sequences: List of sequences to cluster
        identity_threshold: Minimum sequence identity (default 0.7)

    Returns:
        Dictionary mapping cluster_id -> [sequence_indices]
    """
    # Check if cd-hit is available
    try:
        subprocess.run(
            ['cd-hit', '-h'],
            capture_output=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "CD-HIT not found. Please install: "
            "conda install -c bioconda cd-hit"
        )

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write sequences to FASTA
        fasta_file = tmpdir_path / "sequences.fasta"
        with open(fasta_file, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq_{i}\n{seq}\n")

        # Run CD-HIT
        output_file = tmpdir_path / "clusters"
        subprocess.run([
            'cd-hit',
            '-i', str(fasta_file),
            '-o', str(output_file),
            '-c', str(identity_threshold),
            '-n', '5',  # Word length
            '-M', '2000',  # Memory limit (MB)
            '-T', '1',  # Number of threads
        ], capture_output=True, check=True)

        # Parse cluster file
        cluster_file = str(output_file) + ".clstr"
        clusters = defaultdict(list)
        current_cluster = -1

        with open(cluster_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>Cluster'):
                    current_cluster += 1
                elif line:
                    # Parse sequence ID
                    # Format: >seq_123... at 95%
                    seq_id = int(line.split('>seq_')[1].split('.')[0])
                    clusters[current_cluster].append(seq_id)

    return dict(clusters)


def cluster_sequences(
    sequences: List[str],
    identity_threshold: float = 0.7,
    method: str = 'auto',
) -> Dict[int, List[int]]:
    """
    Cluster sequences using available method.

    Tries MMseqs2 first, falls back to CD-HIT, then simple pairwise.

    Args:
        sequences: List of sequences to cluster
        identity_threshold: Minimum sequence identity (default 0.7)
        method: 'auto', 'mmseqs2', 'cdhit', or 'simple'

    Returns:
        Dictionary mapping cluster_id -> [sequence_indices]
    """
    if method == 'mmseqs2':
        return cluster_sequences_mmseqs2(sequences, identity_threshold)
    elif method == 'cdhit':
        return cluster_sequences_cdhit(sequences, identity_threshold)
    elif method == 'simple':
        return cluster_sequences_simple(sequences, identity_threshold)
    elif method == 'auto':
        # Try MMseqs2 first
        try:
            return cluster_sequences_mmseqs2(sequences, identity_threshold)
        except RuntimeError:
            pass

        # Try CD-HIT
        try:
            return cluster_sequences_cdhit(sequences, identity_threshold)
        except RuntimeError:
            pass

        # Fall back to simple pairwise
        return cluster_sequences_simple(sequences, identity_threshold)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def cluster_sequences_simple(
    sequences: List[str],
    identity_threshold: float = 0.7,
) -> Dict[int, List[int]]:
    """
    Simple greedy clustering using pairwise alignment.

    Warning: O(n²) complexity, only for small datasets.

    Args:
        sequences: List of sequences to cluster
        identity_threshold: Minimum sequence identity

    Returns:
        Dictionary mapping cluster_id -> [sequence_indices]
    """
    n = len(sequences)
    if n > 500:
        raise ValueError(
            f"Simple clustering not recommended for {n} sequences (>500). "
            "Install MMseqs2 or CD-HIT."
        )

    # Initialize clusters
    clusters = {}
    seq_to_cluster = {}
    next_cluster_id = 0

    for i in range(n):
        if i in seq_to_cluster:
            continue

        # Start new cluster
        cluster_id = next_cluster_id
        next_cluster_id += 1
        clusters[cluster_id] = [i]
        seq_to_cluster[i] = cluster_id

        # Find similar sequences
        for j in range(i + 1, n):
            if j in seq_to_cluster:
                continue

            similarity = compute_sequence_similarity(sequences[i], sequences[j])
            if similarity >= identity_threshold:
                clusters[cluster_id].append(j)
                seq_to_cluster[j] = cluster_id

    return clusters


def identify_derivative_groups(
    samples: List[PEMSample],
    max_edit_distance: int = 1,
) -> Dict[int, List[int]]:
    """
    Identify groups of same-scaffold derivatives.

    Derivatives are defined as samples with:
    - Same backbone sequence
    - Edit distance ≤ max_edit_distance

    Args:
        samples: List of PEMSample objects
        max_edit_distance: Maximum edit distance to be considered derivatives

    Returns:
        Dictionary mapping group_id -> [sample_indices]
    """
    # Group by sequence
    seq_groups = defaultdict(list)
    for i, sample in enumerate(samples):
        seq_groups[sample.sequence].append(i)

    # For each sequence group, cluster by edit similarity
    derivative_groups = {}
    group_id = 0

    for seq, indices in seq_groups.items():
        if len(indices) == 1:
            # Singleton, not a derivative group
            continue

        # Extract edit signatures
        edit_signatures = []
        for idx in indices:
            sample = samples[idx]
            # Signature: set of (edit_type, positions_tuple)
            sig = frozenset(
                (edit.edit_type, tuple(sorted(edit.anchor_positions)))
                for edit in sample.edits
            )
            edit_signatures.append(sig)

        # Cluster by edit distance
        assigned = set()
        for i, idx_i in enumerate(indices):
            if i in assigned:
                continue

            # Start new group
            current_group = [idx_i]
            assigned.add(i)

            # Find similar edits
            for j, idx_j in enumerate(indices):
                if j in assigned or j <= i:
                    continue

                # Compute edit distance (symmetric difference size)
                dist = len(edit_signatures[i] ^ edit_signatures[j])

                if dist <= max_edit_distance:
                    current_group.append(idx_j)
                    assigned.add(j)

            # Only keep as derivative group if >1 member
            if len(current_group) > 1:
                derivative_groups[group_id] = current_group
                group_id += 1

    return derivative_groups


def get_edit_profile(sample: PEMSample) -> str:
    """
    Get edit profile for a sample.

    Profile is a string describing the types of edits present.

    Args:
        sample: PEMSample object

    Returns:
        Edit profile string
    """
    if not sample.edits:
        return "no_edits"

    # Get unique edit families
    families = sorted(set(edit.edit_family for edit in sample.edits))
    return "+".join(families)


def compute_edit_profile_distribution(
    samples: List[PEMSample]
) -> Dict[str, int]:
    """
    Compute distribution of edit profiles.

    Args:
        samples: List of PEMSample objects

    Returns:
        Dictionary mapping profile -> count
    """
    distribution = defaultdict(int)
    for sample in samples:
        profile = get_edit_profile(sample)
        distribution[profile] += 1
    return dict(distribution)
