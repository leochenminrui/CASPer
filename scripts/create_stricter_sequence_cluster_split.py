#!/usr/bin/env python3
"""
Create stricter sequence-cluster-based split (50% identity threshold).

This is a harder generalization test than the 70% identity split,
ensuring even greater sequence dissimilarity between train and test sets.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from collections import defaultdict
from typing import List, Dict
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.pem_schema import PEMSample
from data.serialization import save_jsonl, load_samples


def compute_sequence_similarity(seq1: str, seq2: str) -> float:
    """Compute simple sequence identity."""
    if len(seq1) != len(seq2):
        return 0.0
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)


def cluster_sequences_by_identity(sequences: List[str], threshold: float = 0.5) -> Dict[int, List[int]]:
    """
    Cluster sequences by pairwise identity.

    Returns dict mapping cluster_id -> list of sequence indices
    """
    n = len(sequences)
    clusters = {}
    cluster_id = 0
    assigned = set()

    for i in range(n):
        if i in assigned:
            continue

        # Start new cluster
        cluster = [i]
        assigned.add(i)

        # Find similar sequences
        for j in range(i+1, n):
            if j in assigned:
                continue

            # Check similarity to cluster representative
            sim = compute_sequence_similarity(sequences[i], sequences[j])
            if sim >= threshold:
                cluster.append(j)
                assigned.add(j)

        clusters[cluster_id] = cluster
        cluster_id += 1

    return clusters


def create_cluster_based_split(
    samples: List[PEMSample],
    identity_threshold: float = 0.5,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[PEMSample]]:
    """Create sequence-cluster-based split."""

    np.random.seed(seed)

    # Extract unique sequences
    seq_to_samples = defaultdict(list)
    for sample in samples:
        seq_to_samples[sample.sequence].append(sample)

    unique_sequences = list(seq_to_samples.keys())
    print(f"Unique sequences: {len(unique_sequences)}")

    # Cluster sequences by similarity
    print(f"Clustering sequences (threshold={identity_threshold})...")
    clusters = cluster_sequences_by_identity(unique_sequences, threshold=identity_threshold)
    print(f"Number of clusters: {len(clusters)}")

    # Get cluster sizes (in terms of total samples, not just unique sequences)
    cluster_sample_counts = []
    for cid, seq_indices in clusters.items():
        total_samples = sum(len(seq_to_samples[unique_sequences[idx]]) for idx in seq_indices)
        cluster_sample_counts.append((cid, total_samples, seq_indices))

    cluster_sample_counts.sort(key=lambda x: -x[1])

    print(f"Largest cluster: {cluster_sample_counts[0][1]} samples")
    print(f"Smallest cluster: {cluster_sample_counts[-1][1]} samples")

    # Assign clusters to splits
    # Goal: test ~15%, val ~15%, train ~70% by sample count
    total_samples = len(samples)
    target_test = total_samples * test_ratio
    target_val = total_samples * val_ratio

    test_clusters = []
    val_clusters = []
    train_clusters = []

    test_count = 0
    val_count = 0

    # Shuffle clusters
    np.random.shuffle(cluster_sample_counts)

    for cid, sample_count, seq_indices in cluster_sample_counts:
        if test_count < target_test:
            test_clusters.append((cid, seq_indices))
            test_count += sample_count
        elif val_count < target_val:
            val_clusters.append((cid, seq_indices))
            val_count += sample_count
        else:
            train_clusters.append((cid, seq_indices))

    print(f"\nCluster assignment:")
    print(f"  Train: {len(train_clusters)} clusters")
    print(f"  Val: {len(val_clusters)} clusters")
    print(f"  Test: {len(test_clusters)} clusters")

    # Collect samples
    train_samples = []
    val_samples = []
    test_samples = []

    for cid, seq_indices in train_clusters:
        for seq_idx in seq_indices:
            seq = unique_sequences[seq_idx]
            train_samples.extend(seq_to_samples[seq])

    for cid, seq_indices in val_clusters:
        for seq_idx in seq_indices:
            seq = unique_sequences[seq_idx]
            val_samples.extend(seq_to_samples[seq])

    for cid, seq_indices in test_clusters:
        for seq_idx in seq_indices:
            seq = unique_sequences[seq_idx]
            test_samples.extend(seq_to_samples[seq])

    print(f"\nSample counts:")
    print(f"  Train: {len(train_samples)} ({100*len(train_samples)/total_samples:.1f}%)")
    print(f"  Val: {len(val_samples)} ({100*len(val_samples)/total_samples:.1f}%)")
    print(f"  Test: {len(test_samples)} ({100*len(test_samples)/total_samples:.1f}%)")

    return {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }


def main():
    parser = argparse.ArgumentParser(description="Create stricter sequence-cluster split (50% identity)")
    parser.add_argument('--dataset', type=str, default='CycPeptMPDB_PAMPA',
                        help='Dataset name')
    parser.add_argument('--identity-threshold', type=float, default=0.5,
                        help='Sequence identity threshold (default: 0.5 for 50%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    print("="*80)
    print(f"STRICTER SEQUENCE-CLUSTER SPLIT ({int(args.identity_threshold*100)}% identity)")
    print("="*80)

    # Load dataset
    data_path = Path(f"data/processed/pem_schema/{args.dataset.lower()}.jsonl")

    if not data_path.exists():
        print(f"Error: Dataset file not found: {data_path}")
        return

    print(f"\nLoading dataset from {data_path}...")
    samples = load_samples(data_path)
    print(f"Total samples: {len(samples)}")

    # Create split
    print(f"\nCreating sequence-cluster split (identity threshold = {args.identity_threshold})...")
    splits = create_cluster_based_split(
        samples,
        identity_threshold=args.identity_threshold,
        seed=args.seed
    )

    # Save splits
    split_name = f"sequence_cluster_{int(args.identity_threshold*100)}pct"
    output_dir = Path(f"data/splits/{args.dataset}/{split_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving splits to {output_dir}...")

    for split_type, split_samples in splits.items():
        output_file = output_dir / f"{split_type}.jsonl"
        save_jsonl(split_samples, output_file)
        print(f"  Saved {split_type}: {len(split_samples)} samples")

    # Save split info
    split_info = {
        'dataset': args.dataset,
        'split_type': split_name,
        'identity_threshold': args.identity_threshold,
        'seed': args.seed,
        'n_train': len(splits['train']),
        'n_val': len(splits['val']),
        'n_test': len(splits['test']),
        'n_total': len(samples)
    }

    info_file = output_dir / "split_info.json"
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\n✅ Split created successfully!")
    print(f"   Output directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
