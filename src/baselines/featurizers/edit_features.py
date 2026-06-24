"""
Edit-based feature extractors.

Features derived from chemical edits including counts, families, and anchor windows.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter

from data.pem_schema import PEMSample


class EditCountFeaturizer:
    """
    Simple edit count features.

    Counts number of edits, edit families, and edit types.
    """

    def __init__(self, include_family_counts: bool = True):
        """
        Initialize edit count featurizer.

        Args:
            include_family_counts: Whether to include per-family counts
        """
        self.include_family_counts = include_family_counts
        self.known_families = set()  # Populated during featurization

    def featurize_sample(self, sample: PEMSample) -> np.ndarray:
        """
        Featurize a single sample.

        Args:
            sample: PEMSample with edits

        Returns:
            Feature vector
        """
        features = []

        # Basic counts
        features.append(len(sample.edits))  # Total number of edits

        # Family counts
        family_counts = Counter(e.edit_family for e in sample.edits)
        features.append(len(family_counts))  # Number of unique families

        # Per-family counts (if enabled)
        if self.include_family_counts:
            # Update known families
            self.known_families.update(family_counts.keys())

            # Create one-hot-like counts for each family
            # Note: This requires seeing all data first in practice
            for family in sorted(self.known_families):
                features.append(family_counts.get(family, 0))

        return np.array(features, dtype=np.float32)

    def featurize(self, samples: List[PEMSample]) -> np.ndarray:
        """
        Featurize multiple samples.

        Args:
            samples: List of PEMSample objects

        Returns:
            Feature matrix (n_samples, n_features)
        """
        # First pass: collect all families
        if self.include_family_counts:
            for sample in samples:
                for edit in sample.edits:
                    self.known_families.add(edit.edit_family)

        # Second pass: featurize
        feature_vectors = []
        for sample in samples:
            feat = self.featurize_sample(sample)
            feature_vectors.append(feat)

        return np.array(feature_vectors)

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = ['num_edits', 'num_edit_families']

        if self.include_family_counts:
            for family in sorted(self.known_families):
                names.append(f'count_{family}')

        return names


class AnchorWindowFeaturizer:
    """
    Anchor window features.

    Extracts features from local windows around modification sites.
    Uses anchor positions but only for feature extraction (not learnable).
    """

    def __init__(
        self,
        window_size: int = 2,
        include_aa_properties: bool = True,
        include_descriptors: bool = False,
    ):
        """
        Initialize anchor window featurizer.

        Args:
            window_size: Number of residues on each side of anchor
            include_aa_properties: Include AA property features
            include_descriptors: Include chemical descriptors
        """
        self.window_size = window_size
        self.include_aa_properties = include_aa_properties
        self.include_descriptors = include_descriptors

        # AA properties (hydrophobicity, charge, etc.)
        self.aa_properties = self._get_aa_properties()

    def _get_aa_properties(self) -> Dict[str, np.ndarray]:
        """Get amino acid properties."""
        # Simple property encoding (hydrophobicity, charge, size)
        properties = {
            'A': [1.8, 0, 0],   # Alanine: hydrophobic, neutral, small
            'R': [-4.5, 1, 1],  # Arginine: hydrophilic, positive, large
            'N': [-3.5, 0, 0],  # Asparagine
            'D': [-3.5, -1, 0], # Aspartic acid: acidic, negative
            'C': [2.5, 0, 0],   # Cysteine
            'Q': [-3.5, 0, 0],  # Glutamine
            'E': [-3.5, -1, 0], # Glutamic acid: acidic, negative
            'G': [-0.4, 0, 0],  # Glycine: small
            'H': [-3.2, 0.5, 0], # Histidine: can be positive
            'I': [4.5, 0, 1],   # Isoleucine: hydrophobic, large
            'L': [3.8, 0, 1],   # Leucine: hydrophobic, large
            'K': [-3.9, 1, 0],  # Lysine: positive
            'M': [1.9, 0, 0],   # Methionine
            'F': [2.8, 0, 1],   # Phenylalanine: aromatic, large
            'P': [-1.6, 0, 0],  # Proline
            'S': [-0.8, 0, 0],  # Serine
            'T': [-0.7, 0, 0],  # Threonine
            'W': [-0.9, 0, 1],  # Tryptophan: aromatic, large
            'Y': [-1.3, 0, 1],  # Tyrosine: aromatic
            'V': [4.2, 0, 0],   # Valine: hydrophobic
        }

        # Convert to numpy arrays
        return {aa: np.array(props, dtype=np.float32) for aa, props in properties.items()}

    def extract_window_features(
        self,
        sequence: str,
        anchor_position: int,
    ) -> np.ndarray:
        """
        Extract features from window around anchor.

        Args:
            sequence: Protein sequence
            anchor_position: Position of modification (0-indexed)

        Returns:
            Window feature vector
        """
        features = []

        # Extract window
        start = max(0, anchor_position - self.window_size)
        end = min(len(sequence), anchor_position + self.window_size + 1)
        window = sequence[start:end]

        # Pad if needed
        if anchor_position < self.window_size:
            window = 'X' * (self.window_size - anchor_position) + window
        if anchor_position + self.window_size >= len(sequence):
            window = window + 'X' * (anchor_position + self.window_size - len(sequence) + 1)

        # AA properties for each position
        if self.include_aa_properties:
            for aa in window:
                if aa in self.aa_properties:
                    features.extend(self.aa_properties[aa])
                else:
                    features.extend([0, 0, 0])  # Unknown AA

        # Could add more features here (position in sequence, etc.)
        features.append(anchor_position / max(len(sequence), 1))  # Normalized position

        return np.array(features, dtype=np.float32)

    def featurize_sample(self, sample: PEMSample) -> np.ndarray:
        """
        Featurize a single sample.

        Args:
            sample: PEMSample with edits

        Returns:
            Feature vector (aggregated across all modifications)
        """
        window_features = []

        for edit in sample.edits:
            # Extract window features for this modification
            window_feat = self.extract_window_features(
                sample.sequence,
                edit.anchor_position,
            )
            window_features.append(window_feat)

        # Aggregate window features
        if len(window_features) == 0:
            # No modifications - return zeros
            dummy_size = (2 * self.window_size + 1) * 3 + 1  # AA properties + position
            return np.zeros(dummy_size, dtype=np.float32)

        window_features = np.array(window_features)

        # Use mean aggregation
        aggregated = np.mean(window_features, axis=0)

        # Optionally add chemical descriptors
        if self.include_descriptors:
            from .descriptors import ChemicalDescriptorFeaturizer
            desc_featurizer = ChemicalDescriptorFeaturizer(aggregation="mean")
            desc_features = desc_featurizer.featurize_sample(sample)
            aggregated = np.concatenate([aggregated, desc_features])

        return aggregated

    def featurize(self, samples: List[PEMSample]) -> np.ndarray:
        """
        Featurize multiple samples.

        Args:
            samples: List of PEMSample objects

        Returns:
            Feature matrix (n_samples, n_features)
        """
        feature_vectors = []
        for sample in samples:
            feat = self.featurize_sample(sample)
            feature_vectors.append(feat)

        return np.array(feature_vectors)

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []

        # Window position names
        for i in range(-self.window_size, self.window_size + 1):
            if self.include_aa_properties:
                names.extend([
                    f'pos{i}_hydrophobicity',
                    f'pos{i}_charge',
                    f'pos{i}_size',
                ])

        names.append('normalized_position')

        return names
