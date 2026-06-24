#!/usr/bin/env python3
"""
Generate 70% sequence-cluster split for CycPeptMPDB PAMPA.

Uses Biopython global alignment for accurate sequence clustering.
Split-by-cluster ensures no test sequence has >70% identity to any train sequence.

Run from project root:
    python scripts/create_sequence_cluster_split.py
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pem_schema import PEMSample
from src.data.serialization import load_jsonl, save_jsonl


def cluster_sequences(sequences, ids, identity_threshold=0.70):
    """Greedy clustering with Biopython global alignment."""
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5

    n = len(sequences)
    logger.info(f"Clustering {n} unique sequences at {identity_threshold:.0%} identity...")

    # Sort by length descending
    order = sorted(range(n), key=lambda i: len(sequences[i]), reverse=True)

    clusters = {}
    assigned = set()
    next_cluster_id = 0
    total_alignments = 0

    for i_idx, i in enumerate(order):
        if i in assigned:
            continue

        if (i_idx + 1) % 200 == 0:
            logger.info(f"  {i_idx+1}/{n} ({len(assigned)} assigned, "
                        f"{next_cluster_id} clusters, {total_alignments} alignments)")

        cluster_id = next_cluster_id
        next_cluster_id += 1
        clusters[cluster_id] = [ids[i]]
        assigned.add(i)

        seq_i = sequences[i]
        len_i = len(seq_i)

        for j_idx in range(i_idx + 1, n):
            j = order[j_idx]
            if j in assigned:
                continue

            seq_j = sequences[j]
            len_j = len(seq_j)

            # Quick length-based pre-filter: can't be >70% identical if lengths differ by >30%
            max_len, min_len = max(len_i, len_j), min(len_i, len_j)
            if max_len == 0 or min_len / max_len < 0.40:
                continue

            # Full alignment
            total_alignments += 1
            try:
                aln = aligner.align(seq_i, seq_j)
                if aln:
                    score = aln[0].score
                    max_possible = max(len_i, len_j)
                    identity = score / max_possible if max_possible > 0 else 0
                    if identity >= identity_threshold:
                        clusters[cluster_id].append(ids[j])
                        assigned.add(j)
            except Exception:
                pass

    # Handle unassigned singletons
    for i in range(n):
        if i not in assigned:
            clusters[next_cluster_id] = [ids[i]]
            next_cluster_id += 1
            assigned.add(i)

    logger.info(f"Clustering done: {len(clusters)} clusters, {total_alignments} alignments")
    return clusters


def main():
    # Load samples
    input_file = PROJECT_ROOT / "data/processed/pem_schema/cycpeptmpdb_pampa.jsonl"
    logger.info(f"Loading: {input_file}")
    all_samples = load_jsonl(input_file)
    logger.info(f"Loaded {len(all_samples)} samples")

    # Extract unique sequences
    seq_to_ids = defaultdict(list)
    for s in all_samples:
        seq_to_ids[s.sequence].append(s.sample_id)

    unique_seqs = list(seq_to_ids.keys())
    rep_ids = [seq_to_ids[seq][0] for seq in unique_seqs]
    logger.info(f"Unique sequences: {len(unique_seqs)}")

    # Cluster unique sequences
    seq_clusters = cluster_sequences(unique_seqs, rep_ids, identity_threshold=0.70)

    # Expand clusters: all samples sharing same sequence go to same cluster
    sample_to_cluster = {}
    for cl_id, members in seq_clusters.items():
        for member_id in members:
            # Find the sequence for this member
            for seq, ids in seq_to_ids.items():
                if member_id in ids:
                    for sid in ids:
                        sample_to_cluster[sid] = cl_id
                    break

    # Group samples by cluster
    cluster_to_samples = defaultdict(list)
    for i, s in enumerate(all_samples):
        cid = sample_to_cluster.get(s.sample_id, f"singleton_{i}")
        cluster_to_samples[cid].append(i)

    cluster_ids = list(cluster_to_samples.keys())
    n_clusters = len(cluster_ids)
    logger.info(f"Expanded to {len(all_samples)} samples in {n_clusters} clusters")

    # Split by cluster
    rng = np.random.RandomState(42)
    rng.shuffle(cluster_ids)

    # Greedy assignment to match 70/15/15 ratios
    total = len(all_samples)
    target_test = int(total * 0.15)
    target_val = int(total * 0.15)

    train_clusters, val_clusters, test_clusters = [], [], []
    test_size, val_size = 0, 0

    for cid in cluster_ids:
        n = len(cluster_to_samples[cid])
        if test_size < target_test:
            test_clusters.append(cid)
            test_size += n
        elif val_size < target_val:
            val_clusters.append(cid)
            val_size += n
        else:
            train_clusters.append(cid)

    # Collect samples
    def collect(cl_list):
        result = []
        for cid in cl_list:
            for idx in cluster_to_samples[cid]:
                result.append(all_samples[idx])
        rng.shuffle(result)
        return result

    train = collect(train_clusters)
    val = collect(val_clusters)
    test = collect(test_clusters)

    logger.info(f"Split: train={len(train)} ({100*len(train)/total:.1f}%), "
                f"val={len(val)} ({100*len(val)/total:.1f}%), "
                f"test={len(test)} ({100*len(test)/total:.1f}%)")

    # Save
    output_dir = PROJECT_ROOT / "data/splits/CycPeptMPDB_PAMPA/sequence_cluster"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, samples in [("train", train), ("val", val), ("test", test)]:
        save_jsonl(samples, output_dir / f"{name}.jsonl")

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump({
            'strategy': 'sequence_cluster',
            'identity_threshold': 0.70,
            'n_total': len(all_samples),
            'n_unique_sequences': len(unique_seqs),
            'n_clusters': n_clusters,
            'train': len(train), 'val': len(val), 'test': len(test),
            'n_train_clusters': len(train_clusters),
            'n_val_clusters': len(val_clusters),
            'n_test_clusters': len(test_clusters),
        }, f, indent=2)

    logger.info(f"Cluster split saved to: {output_dir}")


if __name__ == "__main__":
    main()
