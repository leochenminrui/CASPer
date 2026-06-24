"""
Leakage analysis for data splits.

Computes various metrics to quantify data leakage risk between
train/val/test splits.
"""

from typing import List, Dict, Set, Tuple
import numpy as np
from scipy import stats
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from ..pem_schema import PEMSample
from .base import LeakageAnalysis, LeakageRisk
from .utils import (
    compute_sequence_similarity,
    extract_scaffold,
    get_edit_profile,
)


def compute_sequence_similarity_matrix(
    sequences1: List[str],
    sequences2: List[str],
    max_comparisons: int = 10000,
) -> np.ndarray:
    """
    Compute pairwise sequence similarity matrix.

    For large datasets, samples randomly to avoid O(n²) explosion.

    Args:
        sequences1: First set of sequences
        sequences2: Second set of sequences
        max_comparisons: Maximum number of comparisons (default 10000)

    Returns:
        Matrix of similarities (n1 x n2) or sampled subset
    """
    n1 = len(sequences1)
    n2 = len(sequences2)
    total_comparisons = n1 * n2

    # If too many comparisons, sample
    if total_comparisons > max_comparisons:
        # Sample pairs
        n_samples = min(max_comparisons, total_comparisons)
        rng = np.random.RandomState(42)

        similarities = []
        for _ in range(n_samples):
            i = rng.randint(0, n1)
            j = rng.randint(0, n2)
            sim = compute_sequence_similarity(sequences1[i], sequences2[j])
            similarities.append(sim)

        return np.array(similarities)

    # Compute full matrix
    similarities = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            similarities[i, j] = compute_sequence_similarity(
                sequences1[i], sequences2[j]
            )

    return similarities


def check_scaffold_overlap(
    train_samples: List[PEMSample],
    test_samples: List[PEMSample],
) -> Tuple[int, float]:
    """
    Check for scaffold overlap between train and test.

    Args:
        train_samples: Training samples
        test_samples: Test samples

    Returns:
        (n_shared_scaffolds, pct_test_with_shared_scaffold)
    """
    train_scaffolds = {extract_scaffold(s) for s in train_samples}
    test_scaffolds = {extract_scaffold(s) for s in test_samples}

    shared = train_scaffolds & test_scaffolds
    n_shared = len(shared)

    # Count test samples with shared scaffold
    test_with_shared = sum(
        1 for s in test_samples
        if extract_scaffold(s) in shared
    )
    pct_shared = (test_with_shared / len(test_samples)) * 100 if test_samples else 0.0

    return n_shared, pct_shared


def compute_edit_profile_vectors(
    samples: List[PEMSample]
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute edit profile feature vectors for samples.

    Args:
        samples: List of PEMSample objects

    Returns:
        (feature_matrix, profile_names)
        feature_matrix: n_samples x n_profiles binary matrix
        profile_names: List of profile names
    """
    # Get all profiles
    all_profiles = sorted(set(get_edit_profile(s) for s in samples))

    # Create binary vectors
    vectors = np.zeros((len(samples), len(all_profiles)))
    for i, sample in enumerate(samples):
        profile = get_edit_profile(sample)
        j = all_profiles.index(profile)
        vectors[i, j] = 1

    return vectors, all_profiles


def compute_edit_profile_similarity(
    train_samples: List[PEMSample],
    test_samples: List[PEMSample],
) -> float:
    """
    Compute cosine similarity of edit profile distributions.

    Args:
        train_samples: Training samples
        test_samples: Test samples

    Returns:
        Cosine similarity (0.0 to 1.0)
    """
    # Get profile distributions
    train_profiles = [get_edit_profile(s) for s in train_samples]
    test_profiles = [get_edit_profile(s) for s in test_samples]

    # Get all profiles
    all_profiles = sorted(set(train_profiles + test_profiles))

    # Create frequency vectors
    train_vec = np.array([train_profiles.count(p) for p in all_profiles])
    test_vec = np.array([test_profiles.count(p) for p in all_profiles])

    # Normalize
    train_vec = train_vec / train_vec.sum() if train_vec.sum() > 0 else train_vec
    test_vec = test_vec / test_vec.sum() if test_vec.sum() > 0 else test_vec

    # Compute cosine similarity
    similarity = cosine_similarity(
        train_vec.reshape(1, -1),
        test_vec.reshape(1, -1)
    )[0, 0]

    return float(similarity)


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.

    Args:
        p: First distribution
        q: Second distribution

    Returns:
        JS divergence (0.0 to 1.0)
    """
    # Normalize
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    p = p / p.sum()
    q = q / q.sum()

    # Compute M = (P + Q) / 2
    m = (p + q) / 2

    # Compute JS divergence
    kl_pm = stats.entropy(p, m)
    kl_qm = stats.entropy(q, m)
    js = (kl_pm + kl_qm) / 2

    return float(js)


def compute_edit_family_js_divergence(
    train_samples: List[PEMSample],
    test_samples: List[PEMSample],
) -> float:
    """
    Compute JS divergence of edit family distributions.

    Args:
        train_samples: Training samples
        test_samples: Test samples

    Returns:
        JS divergence
    """
    # Get all edit families
    from ..pem_schema import EditFamily
    all_families = [f.value for f in EditFamily]

    # Count families in each split
    train_counts = Counter()
    test_counts = Counter()

    for sample in train_samples:
        for edit in sample.edits:
            train_counts[edit.edit_family] += 1

    for sample in test_samples:
        for edit in sample.edits:
            test_counts[edit.edit_family] += 1

    # Create vectors
    train_vec = np.array([train_counts.get(f, 0) for f in all_families])
    test_vec = np.array([test_counts.get(f, 0) for f in all_families])

    # Compute JS divergence
    return compute_js_divergence(train_vec, test_vec)


def compare_label_distributions(
    train_samples: List[PEMSample],
    val_samples: List[PEMSample],
    test_samples: List[PEMSample],
) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    """
    Compare label distributions across splits using KS test.

    Args:
        train_samples: Training samples
        val_samples: Validation samples
        test_samples: Test samples

    Returns:
        (ks_statistic, ks_pvalue, stats_dict)
        stats_dict: {split_name: (mean, std)}
    """
    # Extract labels
    train_labels = np.array([s.label for s in train_samples])
    val_labels = np.array([s.label for s in val_samples])
    test_labels = np.array([s.label for s in test_samples])

    # KS test (train vs test)
    if len(train_labels) > 0 and len(test_labels) > 0:
        ks_stat, ks_pval = stats.ks_2samp(train_labels, test_labels)
    else:
        ks_stat, ks_pval = 0.0, 1.0

    # Compute statistics
    stats_dict = {
        'train': (
            float(np.mean(train_labels)) if len(train_labels) > 0 else 0.0,
            float(np.std(train_labels)) if len(train_labels) > 0 else 0.0,
        ),
        'val': (
            float(np.mean(val_labels)) if len(val_labels) > 0 else 0.0,
            float(np.std(val_labels)) if len(val_labels) > 0 else 0.0,
        ),
        'test': (
            float(np.mean(test_labels)) if len(test_labels) > 0 else 0.0,
            float(np.std(test_labels)) if len(test_labels) > 0 else 0.0,
        ),
    }

    return float(ks_stat), float(ks_pval), stats_dict


def classify_leakage_risk(
    max_sequence_identity: float,
    pct_test_shared_scaffold: float,
) -> LeakageRisk:
    """
    Classify overall leakage risk.

    Args:
        max_sequence_identity: Maximum sequence identity train->test
        pct_test_shared_scaffold: % of test with shared scaffold

    Returns:
        LeakageRisk classification
    """
    # Low risk: Low sequence similarity AND no scaffold overlap
    if max_sequence_identity < 0.70 and pct_test_shared_scaffold < 10.0:
        return LeakageRisk.LOW

    # High risk: High sequence similarity OR significant scaffold overlap
    if max_sequence_identity > 0.85 or pct_test_shared_scaffold > 50.0:
        return LeakageRisk.HIGH

    # Moderate risk: In between
    return LeakageRisk.MODERATE


def analyze_leakage(
    train_samples: List[PEMSample],
    val_samples: List[PEMSample],
    test_samples: List[PEMSample],
) -> LeakageAnalysis:
    """
    Perform comprehensive leakage analysis.

    Args:
        train_samples: Training samples
        val_samples: Validation samples
        test_samples: Test samples

    Returns:
        LeakageAnalysis object with all metrics
    """
    # Sequence similarity
    train_seqs = [s.sequence for s in train_samples]
    test_seqs = [s.sequence for s in test_samples]

    if train_seqs and test_seqs:
        sim_matrix = compute_sequence_similarity_matrix(train_seqs, test_seqs)
        max_identity = float(np.max(sim_matrix))
        mean_identity = float(np.mean(sim_matrix))

        # Count test sequences with high similarity to train
        if sim_matrix.ndim == 2:
            max_per_test = np.max(sim_matrix, axis=0)
        else:
            max_per_test = sim_matrix

        pct_high_sim = (np.sum(max_per_test > 0.7) / len(max_per_test)) * 100
    else:
        max_identity = 0.0
        mean_identity = 0.0
        pct_high_sim = 0.0

    # Scaffold overlap
    n_shared_scaffolds, pct_shared_scaffold = check_scaffold_overlap(
        train_samples, test_samples
    )

    # Edit pattern similarity
    edit_cosine_sim = compute_edit_profile_similarity(train_samples, test_samples)
    edit_js_div = compute_edit_family_js_divergence(train_samples, test_samples)

    # Label distribution
    ks_stat, ks_pval, label_stats = compare_label_distributions(
        train_samples, val_samples, test_samples
    )

    # Overall risk
    risk = classify_leakage_risk(max_identity, pct_shared_scaffold)

    return LeakageAnalysis(
        max_sequence_identity=max_identity,
        pct_test_high_similarity=float(pct_high_sim),
        mean_test_train_similarity=mean_identity,
        n_shared_scaffolds=n_shared_scaffolds,
        pct_test_shared_scaffold=pct_shared_scaffold,
        edit_profile_cosine_similarity=edit_cosine_sim,
        edit_family_js_divergence=edit_js_div,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        train_label_mean=label_stats['train'][0],
        train_label_std=label_stats['train'][1],
        val_label_mean=label_stats['val'][0],
        val_label_std=label_stats['val'][1],
        test_label_mean=label_stats['test'][0],
        test_label_std=label_stats['test'][1],
        overall_risk=risk,
    )
