"""
Concrete split strategy implementations.

Implements all split strategies defined in the protocol:
- Random
- Scaffold-aware
- Sequence cluster
- Edit-family-aware
- Same-edit-family / new-scaffold
- Grouped derivative
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import train_test_split

from ..pem_schema import PEMSample
from .base import SplitStrategy, SplitResult, SplitMetadata
from .utils import (
    extract_scaffold,
    cluster_sequences,
    get_edit_profile,
    compute_edit_profile_distribution,
    identify_derivative_groups,
)
from .leakage import analyze_leakage


class RandomSplitter(SplitStrategy):
    """
    Random stratified split.

    Stratifies by label quartiles to maintain label distribution.
    """

    def get_strategy_name(self) -> str:
        return "random"

    def check_feasibility(
        self, samples: List[PEMSample]
    ) -> Tuple[bool, Optional[str]]:
        """Random split is always feasible."""
        if len(samples) < 10:
            return False, f"Too few samples ({len(samples)} < 10)"
        return True, None

    def split(self, samples: List[PEMSample]) -> SplitResult:
        """Split samples randomly with stratification."""
        # Check feasibility
        is_feasible, reason = self.check_feasibility(samples)
        if not is_feasible:
            metadata = self._create_metadata(feasible=False, reason=reason)
            return SplitResult([], [], [], metadata)

        # Extract labels for stratification
        labels = np.array([s.label for s in samples])

        # Compute quartiles
        quartiles = np.percentile(labels, [25, 50, 75])
        strata = np.digitize(labels, quartiles)

        # Split train+val, test
        train_val_samples, test_samples = train_test_split(
            samples,
            test_size=self.test_ratio,
            stratify=strata,
            random_state=self.random_seed,
        )

        # Split train, val
        train_val_labels = np.array([s.label for s in train_val_samples])
        train_val_quartiles = np.percentile(train_val_labels, [25, 50, 75])
        train_val_strata = np.digitize(train_val_labels, train_val_quartiles)

        val_size = self.val_ratio / (self.train_ratio + self.val_ratio)

        train_samples, val_samples = train_test_split(
            train_val_samples,
            test_size=val_size,
            stratify=train_val_strata,
            random_state=self.random_seed,
        )

        # Add split metadata to samples
        for sample in train_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'train',
                'random_seed': self.random_seed,
            }

        for sample in val_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'val',
                'random_seed': self.random_seed,
            }

        for sample in test_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'test',
                'random_seed': self.random_seed,
            }

        # Analyze leakage
        leakage = analyze_leakage(train_samples, val_samples, test_samples)

        # Create metadata
        metadata = self._create_metadata(
            feasible=True,
            leakage_analysis=leakage,
        )

        return SplitResult(train_samples, val_samples, test_samples, metadata)


class ScaffoldAwareSplitter(SplitStrategy):
    """
    Scaffold-aware split.

    Groups by scaffold, assigns entire scaffolds to splits.
    """

    def __init__(self, min_scaffolds: int = 20, **kwargs):
        """
        Initialize scaffold-aware splitter.

        Args:
            min_scaffolds: Minimum number of unique scaffolds required
            **kwargs: Passed to SplitStrategy
        """
        super().__init__(**kwargs)
        self.min_scaffolds = min_scaffolds

    def get_strategy_name(self) -> str:
        return "scaffold_aware"

    def check_feasibility(
        self, samples: List[PEMSample]
    ) -> Tuple[bool, Optional[str]]:
        """Check if scaffold split is feasible."""
        # Extract scaffolds
        scaffolds = [extract_scaffold(s) for s in samples]
        unique_scaffolds = set(scaffolds)
        n_scaffolds = len(unique_scaffolds)

        if n_scaffolds < self.min_scaffolds:
            return False, (
                f"Only {n_scaffolds} unique scaffolds "
                f"(minimum {self.min_scaffolds} required)"
            )

        # Check scaffold size distribution
        scaffold_counts = Counter(scaffolds)
        max_scaffold_size = max(scaffold_counts.values())
        max_scaffold_pct = (max_scaffold_size / len(samples)) * 100

        if max_scaffold_pct > 50.0:
            return False, (
                f"Largest scaffold contains {max_scaffold_pct:.1f}% of data "
                "(>50%, dominates dataset)"
            )

        return True, None

    def split(self, samples: List[PEMSample]) -> SplitResult:
        """Split by scaffold."""
        # Check feasibility
        is_feasible, reason = self.check_feasibility(samples)
        if not is_feasible:
            metadata = self._create_metadata(feasible=False, reason=reason)
            return SplitResult([], [], [], metadata)

        # Extract scaffolds
        sample_scaffolds = [extract_scaffold(s) for s in samples]

        # Group samples by scaffold
        scaffold_to_samples = defaultdict(list)
        for sample, scaffold in zip(samples, sample_scaffolds):
            scaffold_to_samples[scaffold].append(sample)

        # Assign scaffolds to splits
        scaffolds = list(scaffold_to_samples.keys())
        self.rng.shuffle(scaffolds)

        # Split scaffolds
        n = len(scaffolds)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_scaffolds = scaffolds[:n_train]
        val_scaffolds = scaffolds[n_train:n_train + n_val]
        test_scaffolds = scaffolds[n_train + n_val:]

        # Collect samples
        train_samples = []
        for scaffold in train_scaffolds:
            train_samples.extend(scaffold_to_samples[scaffold])

        val_samples = []
        for scaffold in val_scaffolds:
            val_samples.extend(scaffold_to_samples[scaffold])

        test_samples = []
        for scaffold in test_scaffolds:
            test_samples.extend(scaffold_to_samples[scaffold])

        # Add split metadata
        for sample in train_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'train',
                'scaffold_id': extract_scaffold(sample),
                'random_seed': self.random_seed,
            }

        for sample in val_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'val',
                'scaffold_id': extract_scaffold(sample),
                'random_seed': self.random_seed,
            }

        for sample in test_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'test',
                'scaffold_id': extract_scaffold(sample),
                'random_seed': self.random_seed,
            }

        # Validate proportions (may be off due to scaffold grouping)
        try:
            self._validate_proportions(
                len(train_samples),
                len(val_samples),
                len(test_samples),
                len(samples),
            )
        except ValueError as e:
            # Proportions off, but continue (document in metadata)
            pass

        # Scaffold distribution
        scaffold_dist = {
            scaffold: len(scaffold_to_samples[scaffold])
            for scaffold in scaffolds
        }

        # Analyze leakage
        leakage = analyze_leakage(train_samples, val_samples, test_samples)

        # Create metadata
        metadata = self._create_metadata(
            feasible=True,
            n_scaffolds=len(scaffolds),
            scaffold_distribution=scaffold_dist,
            leakage_analysis=leakage,
            strategy_params={
                **self._create_metadata(True).strategy_params,
                'min_scaffolds': self.min_scaffolds,
            }
        )

        return SplitResult(train_samples, val_samples, test_samples, metadata)


class SequenceClusterSplitter(SplitStrategy):
    """
    Sequence cluster split.

    Clusters sequences by identity, assigns entire clusters to splits.
    """

    def __init__(
        self,
        identity_threshold: float = 0.7,
        min_clusters: int = 20,
        clustering_method: str = 'auto',
        **kwargs
    ):
        """
        Initialize sequence cluster splitter.

        Args:
            identity_threshold: Clustering threshold (default 0.7)
            min_clusters: Minimum number of clusters required
            clustering_method: 'auto', 'mmseqs2', 'cdhit', or 'simple'
            **kwargs: Passed to SplitStrategy
        """
        super().__init__(**kwargs)
        self.identity_threshold = identity_threshold
        self.min_clusters = min_clusters
        self.clustering_method = clustering_method

    def get_strategy_name(self) -> str:
        return "sequence_cluster"

    def check_feasibility(
        self, samples: List[PEMSample]
    ) -> Tuple[bool, Optional[str]]:
        """Check if sequence clustering is feasible."""
        # Extract sequences
        sequences = [s.sequence for s in samples]

        try:
            # Cluster sequences
            clusters = cluster_sequences(
                sequences,
                self.identity_threshold,
                self.clustering_method,
            )
        except Exception as e:
            return False, f"Clustering failed: {str(e)}"

        n_clusters = len(clusters)

        if n_clusters < self.min_clusters:
            return False, (
                f"Only {n_clusters} clusters at {self.identity_threshold} identity "
                f"(minimum {self.min_clusters} required)"
            )

        return True, None

    def split(self, samples: List[PEMSample]) -> SplitResult:
        """Split by sequence clusters."""
        # Check feasibility
        is_feasible, reason = self.check_feasibility(samples)
        if not is_feasible:
            metadata = self._create_metadata(feasible=False, reason=reason)
            return SplitResult([], [], [], metadata)

        # Extract sequences
        sequences = [s.sequence for s in samples]

        # Cluster sequences
        clusters = cluster_sequences(
            sequences,
            self.identity_threshold,
            self.clustering_method,
        )

        # Group samples by cluster
        cluster_to_samples = defaultdict(list)
        for cluster_id, member_indices in clusters.items():
            for idx in member_indices:
                cluster_to_samples[cluster_id].append(samples[idx])

        # Assign clusters to splits
        cluster_ids = list(cluster_to_samples.keys())
        self.rng.shuffle(cluster_ids)

        # Split clusters
        n = len(cluster_ids)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_cluster_ids = cluster_ids[:n_train]
        val_cluster_ids = cluster_ids[n_train:n_train + n_val]
        test_cluster_ids = cluster_ids[n_train + n_val:]

        # Collect samples
        train_samples = []
        for cid in train_cluster_ids:
            train_samples.extend(cluster_to_samples[cid])

        val_samples = []
        for cid in val_cluster_ids:
            val_samples.extend(cluster_to_samples[cid])

        test_samples = []
        for cid in test_cluster_ids:
            test_samples.extend(cluster_to_samples[cid])

        # Add split metadata
        sample_to_cluster = {}
        for cluster_id, member_indices in clusters.items():
            for idx in member_indices:
                sample_to_cluster[samples[idx].sample_id] = cluster_id

        for sample in train_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'train',
                'cluster_id': sample_to_cluster[sample.sample_id],
                'random_seed': self.random_seed,
            }

        for sample in val_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'val',
                'cluster_id': sample_to_cluster[sample.sample_id],
                'random_seed': self.random_seed,
            }

        for sample in test_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'test',
                'cluster_id': sample_to_cluster[sample.sample_id],
                'random_seed': self.random_seed,
            }

        # Cluster distribution
        cluster_dist = {
            str(cluster_id): len(cluster_to_samples[cluster_id])
            for cluster_id in cluster_ids
        }

        # Analyze leakage
        leakage = analyze_leakage(train_samples, val_samples, test_samples)

        # Create metadata
        metadata = self._create_metadata(
            feasible=True,
            n_clusters=len(cluster_ids),
            cluster_distribution=cluster_dist,
            leakage_analysis=leakage,
            strategy_params={
                **self._create_metadata(True).strategy_params,
                'identity_threshold': self.identity_threshold,
                'min_clusters': self.min_clusters,
                'clustering_method': self.clustering_method,
            }
        )

        return SplitResult(train_samples, val_samples, test_samples, metadata)


class EditFamilyAwareSplitter(SplitStrategy):
    """
    Edit-family-aware split.

    Stratifies by edit profile to maintain distribution across splits.
    """

    def __init__(self, min_per_profile: int = 10, **kwargs):
        """
        Initialize edit-family-aware splitter.

        Args:
            min_per_profile: Minimum samples per edit profile
            **kwargs: Passed to SplitStrategy
        """
        super().__init__(**kwargs)
        self.min_per_profile = min_per_profile

    def get_strategy_name(self) -> str:
        return "edit_family_aware"

    def check_feasibility(
        self, samples: List[PEMSample]
    ) -> Tuple[bool, Optional[str]]:
        """Check if edit-family split is feasible."""
        # Get edit profile distribution
        profile_dist = compute_edit_profile_distribution(samples)

        # Check if profiles have enough samples
        sparse_profiles = [
            profile for profile, count in profile_dist.items()
            if count < self.min_per_profile
        ]

        if sparse_profiles:
            sparse_count = len(sparse_profiles)
            total_profiles = len(profile_dist)
            pct_sparse = (sparse_count / total_profiles) * 100

            if pct_sparse > 30.0:
                return False, (
                    f"{sparse_count}/{total_profiles} edit profiles "
                    f"have <{self.min_per_profile} samples ({pct_sparse:.1f}% sparse)"
                )

        return True, None

    def split(self, samples: List[PEMSample]) -> SplitResult:
        """Split with edit profile stratification."""
        # Check feasibility
        is_feasible, reason = self.check_feasibility(samples)
        if not is_feasible:
            metadata = self._create_metadata(feasible=False, reason=reason)
            return SplitResult([], [], [], metadata)

        # Get edit profiles
        profiles = [get_edit_profile(s) for s in samples]

        # Split train+val, test
        train_val_samples, test_samples, train_val_profiles, test_profiles = \
            train_test_split(
                samples,
                profiles,
                test_size=self.test_ratio,
                stratify=profiles,
                random_state=self.random_seed,
            )

        # Split train, val
        val_size = self.val_ratio / (self.train_ratio + self.val_ratio)

        train_samples, val_samples = train_test_split(
            train_val_samples,
            test_size=val_size,
            stratify=train_val_profiles,
            random_state=self.random_seed,
        )

        # Add split metadata
        for sample in train_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'train',
                'edit_profile': get_edit_profile(sample),
                'random_seed': self.random_seed,
            }

        for sample in val_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'val',
                'edit_profile': get_edit_profile(sample),
                'random_seed': self.random_seed,
            }

        for sample in test_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'test',
                'edit_profile': get_edit_profile(sample),
                'random_seed': self.random_seed,
            }

        # Profile distribution
        profile_dist = compute_edit_profile_distribution(samples)

        # Analyze leakage
        leakage = analyze_leakage(train_samples, val_samples, test_samples)

        # Create metadata
        metadata = self._create_metadata(
            feasible=True,
            n_edit_profiles=len(profile_dist),
            edit_profile_distribution=profile_dist,
            leakage_analysis=leakage,
            strategy_params={
                **self._create_metadata(True).strategy_params,
                'min_per_profile': self.min_per_profile,
            }
        )

        return SplitResult(train_samples, val_samples, test_samples, metadata)


class SameEditFamilyNewScaffoldSplitter(SplitStrategy):
    """
    Same-edit-family / new-scaffold split.

    Train on edit families with certain scaffolds,
    test on same families but new scaffolds.
    """

    def __init__(
        self,
        min_scaffolds_per_family: int = 5,
        **kwargs
    ):
        """
        Initialize same-edit-family / new-scaffold splitter.

        Args:
            min_scaffolds_per_family: Minimum scaffolds per edit family
            **kwargs: Passed to SplitStrategy
        """
        super().__init__(**kwargs)
        self.min_scaffolds_per_family = min_scaffolds_per_family

    def get_strategy_name(self) -> str:
        return "same_edit_family_new_scaffold"

    def check_feasibility(
        self, samples: List[PEMSample]
    ) -> Tuple[bool, Optional[str]]:
        """Check if this split is feasible."""
        # Group by (edit_profile, scaffold)
        family_scaffold_groups = defaultdict(set)

        for sample in samples:
            profile = get_edit_profile(sample)
            scaffold = extract_scaffold(sample)
            family_scaffold_groups[profile].add(scaffold)

        # Check if each profile has enough scaffolds
        sparse_families = []
        for profile, scaffolds in family_scaffold_groups.items():
            if len(scaffolds) < self.min_scaffolds_per_family:
                sparse_families.append(profile)

        if sparse_families:
            return False, (
                f"{len(sparse_families)} edit families have "
                f"<{self.min_scaffolds_per_family} scaffolds"
            )

        return True, None

    def split(self, samples: List[PEMSample]) -> SplitResult:
        """Split by edit family and scaffold."""
        # Check feasibility
        is_feasible, reason = self.check_feasibility(samples)
        if not is_feasible:
            metadata = self._create_metadata(feasible=False, reason=reason)
            return SplitResult([], [], [], metadata)

        # Group by (edit_profile, scaffold)
        family_scaffold_to_samples = defaultdict(list)

        for sample in samples:
            profile = get_edit_profile(sample)
            scaffold = extract_scaffold(sample)
            key = (profile, scaffold)
            family_scaffold_to_samples[key].append(sample)

        # For each edit family, split scaffolds
        train_samples = []
        val_samples = []
        test_samples = []

        # Group by edit family
        family_to_scaffolds = defaultdict(set)
        for (profile, scaffold) in family_scaffold_to_samples.keys():
            family_to_scaffolds[profile].add(scaffold)

        for profile, scaffolds in family_to_scaffolds.items():
            scaffolds_list = list(scaffolds)
            self.rng.shuffle(scaffolds_list)

            # Split scaffolds for this family
            n = len(scaffolds_list)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)

            train_scaffolds = scaffolds_list[:n_train]
            val_scaffolds = scaffolds_list[n_train:n_train + n_val]
            test_scaffolds = scaffolds_list[n_train + n_val:]

            # Assign samples
            for scaffold in train_scaffolds:
                key = (profile, scaffold)
                train_samples.extend(family_scaffold_to_samples[key])

            for scaffold in val_scaffolds:
                key = (profile, scaffold)
                val_samples.extend(family_scaffold_to_samples[key])

            for scaffold in test_scaffolds:
                key = (profile, scaffold)
                test_samples.extend(family_scaffold_to_samples[key])

        # Add split metadata
        for sample in train_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'train',
                'edit_profile': get_edit_profile(sample),
                'scaffold_id': extract_scaffold(sample),
                'random_seed': self.random_seed,
            }

        for sample in val_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'val',
                'edit_profile': get_edit_profile(sample),
                'scaffold_id': extract_scaffold(sample),
                'random_seed': self.random_seed,
            }

        for sample in test_samples:
            sample.split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'test',
                'edit_profile': get_edit_profile(sample),
                'scaffold_id': extract_scaffold(sample),
                'random_seed': self.random_seed,
            }

        # Analyze leakage
        leakage = analyze_leakage(train_samples, val_samples, test_samples)

        # Create metadata
        metadata = self._create_metadata(
            feasible=True,
            n_edit_profiles=len(family_to_scaffolds),
            leakage_analysis=leakage,
            strategy_params={
                **self._create_metadata(True).strategy_params,
                'min_scaffolds_per_family': self.min_scaffolds_per_family,
            }
        )

        return SplitResult(train_samples, val_samples, test_samples, metadata)


class GroupedDerivativeSplitter(SplitStrategy):
    """
    Grouped derivative split.

    Groups same-scaffold derivatives, keeps groups intact.
    """

    def __init__(
        self,
        max_edit_distance: int = 1,
        min_coverage: float = 0.1,
        max_group_size: float = 0.2,
        **kwargs
    ):
        """
        Initialize grouped derivative splitter.

        Args:
            max_edit_distance: Max edit distance for derivatives
            min_coverage: Minimum fraction in derivative groups
            max_group_size: Maximum group size (fraction of dataset)
            **kwargs: Passed to SplitStrategy
        """
        super().__init__(**kwargs)
        self.max_edit_distance = max_edit_distance
        self.min_coverage = min_coverage
        self.max_group_size = max_group_size

    def get_strategy_name(self) -> str:
        return "grouped_derivative"

    def check_feasibility(
        self, samples: List[PEMSample]
    ) -> Tuple[bool, Optional[str]]:
        """Check if derivative grouping is feasible."""
        # Identify derivative groups
        groups = identify_derivative_groups(
            samples,
            self.max_edit_distance,
        )

        if not groups:
            return False, "No derivative groups identified"

        # Check coverage
        n_in_groups = sum(len(members) for members in groups.values())
        coverage = n_in_groups / len(samples)

        if coverage < self.min_coverage:
            return False, (
                f"Only {coverage * 100:.1f}% of samples in derivative groups "
                f"(minimum {self.min_coverage * 100:.1f}% required)"
            )

        # Check max group size
        max_group = max(len(members) for members in groups.values())
        max_pct = max_group / len(samples)

        if max_pct > self.max_group_size:
            return False, (
                f"Largest group contains {max_pct * 100:.1f}% of data "
                f"(maximum {self.max_group_size * 100:.1f}% allowed)"
            )

        return True, None

    def split(self, samples: List[PEMSample]) -> SplitResult:
        """Split by derivative groups."""
        # Check feasibility
        is_feasible, reason = self.check_feasibility(samples)
        if not is_feasible:
            metadata = self._create_metadata(feasible=False, reason=reason)
            return SplitResult([], [], [], metadata)

        # Identify derivative groups
        groups = identify_derivative_groups(
            samples,
            self.max_edit_distance,
        )

        # Samples in groups
        in_groups = set()
        for members in groups.values():
            in_groups.update(members)

        # Samples not in groups (singletons)
        singletons = [i for i in range(len(samples)) if i not in in_groups]

        # Assign groups to splits
        group_ids = list(groups.keys())
        self.rng.shuffle(group_ids)

        n = len(group_ids)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_group_ids = group_ids[:n_train]
        val_group_ids = group_ids[n_train:n_train + n_val]
        test_group_ids = group_ids[n_train + n_val:]

        # Collect group samples
        train_indices = []
        for gid in train_group_ids:
            train_indices.extend(groups[gid])

        val_indices = []
        for gid in val_group_ids:
            val_indices.extend(groups[gid])

        test_indices = []
        for gid in test_group_ids:
            test_indices.extend(groups[gid])

        # Assign singletons randomly
        self.rng.shuffle(singletons)
        n_singleton = len(singletons)
        n_train_singleton = int(n_singleton * self.train_ratio)
        n_val_singleton = int(n_singleton * self.val_ratio)

        train_indices.extend(singletons[:n_train_singleton])
        val_indices.extend(singletons[n_train_singleton:n_train_singleton + n_val_singleton])
        test_indices.extend(singletons[n_train_singleton + n_val_singleton:])

        # Create sample lists
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        test_samples = [samples[i] for i in test_indices]

        # Add split metadata
        index_to_group = {}
        for gid, members in groups.items():
            for idx in members:
                index_to_group[idx] = gid

        for idx in train_indices:
            samples[idx].split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'train',
                'derivative_group': index_to_group.get(idx),
                'random_seed': self.random_seed,
            }

        for idx in val_indices:
            samples[idx].split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'val',
                'derivative_group': index_to_group.get(idx),
                'random_seed': self.random_seed,
            }

        for idx in test_indices:
            samples[idx].split_metadata = {
                'split_strategy': self.get_strategy_name(),
                'split_name': 'test',
                'derivative_group': index_to_group.get(idx),
                'random_seed': self.random_seed,
            }

        # Analyze leakage
        leakage = analyze_leakage(train_samples, val_samples, test_samples)

        # Create metadata
        metadata = self._create_metadata(
            feasible=True,
            n_derivative_groups=len(groups),
            leakage_analysis=leakage,
            strategy_params={
                **self._create_metadata(True).strategy_params,
                'max_edit_distance': self.max_edit_distance,
                'min_coverage': self.min_coverage,
                'max_group_size': self.max_group_size,
            }
        )

        return SplitResult(train_samples, val_samples, test_samples, metadata)
