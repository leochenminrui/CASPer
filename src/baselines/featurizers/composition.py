"""
Amino acid composition featurizers.

Simple sequence-only features based on AA composition and basic properties.
"""

from typing import List, Dict
import numpy as np

from data.pem_schema import PEMSample


# Standard amino acids
STANDARD_AA = 'ACDEFGHIKLMNPQRSTVWY'

# Amino acid properties (for grouped composition)
AA_PROPERTIES = {
    'hydrophobic': 'AILMFVPW',
    'polar': 'NQST',
    'positive': 'KRH',
    'negative': 'DE',
    'aromatic': 'FYW',
    'small': 'AGSV',
    'tiny': 'AGS',
    'aliphatic': 'ILV',
}


def compute_aa_composition(sequence: str) -> np.ndarray:
    """
    Compute amino acid composition.

    Args:
        sequence: Amino acid sequence

    Returns:
        Vector of AA frequencies (20-dimensional)
    """
    composition = np.zeros(20)
    sequence = sequence.upper()

    for i, aa in enumerate(STANDARD_AA):
        composition[i] = sequence.count(aa) / len(sequence)

    return composition


def compute_aa_property_composition(sequence: str) -> np.ndarray:
    """
    Compute composition by AA property groups.

    Args:
        sequence: Amino acid sequence

    Returns:
        Vector of property group frequencies
    """
    composition = np.zeros(len(AA_PROPERTIES))
    sequence = sequence.upper()
    seq_len = len(sequence)

    for i, (prop_name, prop_aas) in enumerate(AA_PROPERTIES.items()):
        count = sum(sequence.count(aa) for aa in prop_aas)
        composition[i] = count / seq_len

    return composition


def compute_dipeptide_composition(sequence: str) -> np.ndarray:
    """
    Compute dipeptide composition.

    Args:
        sequence: Amino acid sequence

    Returns:
        Vector of dipeptide frequencies (400-dimensional)
    """
    composition = np.zeros(400)
    sequence = sequence.upper()

    if len(sequence) < 2:
        return composition

    # Create dipeptide index mapping
    dipeptide_to_idx = {}
    idx = 0
    for aa1 in STANDARD_AA:
        for aa2 in STANDARD_AA:
            dipeptide_to_idx[aa1 + aa2] = idx
            idx += 1

    # Count dipeptides
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2]
        if dipeptide in dipeptide_to_idx:
            composition[dipeptide_to_idx[dipeptide]] += 1

    # Normalize
    total = len(sequence) - 1
    if total > 0:
        composition /= total

    return composition


def compute_basic_features(sequence: str) -> np.ndarray:
    """
    Compute basic sequence features.

    Includes:
    - Length
    - Molecular weight (approx)
    - Charge (approx at pH 7)
    - Hydrophobicity (avg)
    - Aromaticity

    Args:
        sequence: Amino acid sequence

    Returns:
        Vector of basic features (5-dimensional)
    """
    # Approximate molecular weights
    MW = {
        'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165,
        'G': 75, 'H': 155, 'I': 131, 'K': 146, 'L': 131,
        'M': 149, 'N': 132, 'P': 115, 'Q': 146, 'R': 174,
        'S': 105, 'T': 119, 'V': 117, 'W': 204, 'Y': 181,
    }

    # Hydrophobicity (Kyte-Doolittle)
    HYDRO = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
    }

    sequence = sequence.upper()

    # Length
    length = len(sequence)

    # Molecular weight
    mw = sum(MW.get(aa, 0) for aa in sequence)

    # Charge (approx at pH 7: +1 for K,R, -1 for D,E)
    charge = (sequence.count('K') + sequence.count('R') -
              sequence.count('D') - sequence.count('E'))

    # Average hydrophobicity
    hydro = sum(HYDRO.get(aa, 0) for aa in sequence) / length if length > 0 else 0

    # Aromaticity
    aromatic = (sequence.count('F') + sequence.count('Y') +
                sequence.count('W')) / length if length > 0 else 0

    return np.array([length, mw, charge, hydro, aromatic])


class AACompositionFeaturizer:
    """
    Featurizer based on amino acid composition.

    Combines multiple composition-based features:
    - AA composition (20D)
    - Property group composition (8D)
    - Basic features (5D)
    - Optionally: dipeptide composition (400D)
    """

    def __init__(
        self,
        use_aa_composition: bool = True,
        use_property_composition: bool = True,
        use_basic_features: bool = True,
        use_dipeptide: bool = False,
    ):
        """
        Initialize featurizer.

        Args:
            use_aa_composition: Include AA composition
            use_property_composition: Include property group composition
            use_basic_features: Include basic features
            use_dipeptide: Include dipeptide composition (increases dim)
        """
        self.use_aa_composition = use_aa_composition
        self.use_property_composition = use_property_composition
        self.use_basic_features = use_basic_features
        self.use_dipeptide = use_dipeptide

        # Compute feature dimension
        self.feature_dim = 0
        if use_aa_composition:
            self.feature_dim += 20
        if use_property_composition:
            self.feature_dim += len(AA_PROPERTIES)
        if use_basic_features:
            self.feature_dim += 5
        if use_dipeptide:
            self.feature_dim += 400

    def featurize_sequence(self, sequence: str) -> np.ndarray:
        """
        Featurize a single sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            Feature vector
        """
        features = []

        if self.use_aa_composition:
            features.append(compute_aa_composition(sequence))

        if self.use_property_composition:
            features.append(compute_aa_property_composition(sequence))

        if self.use_basic_features:
            features.append(compute_basic_features(sequence))

        if self.use_dipeptide:
            features.append(compute_dipeptide_composition(sequence))

        return np.concatenate(features)

    def featurize_samples(self, samples: List[PEMSample]) -> np.ndarray:
        """
        Featurize multiple samples.

        Args:
            samples: List of PEMSample objects

        Returns:
            Feature matrix (n_samples, feature_dim)
        """
        features = np.zeros((len(samples), self.feature_dim))

        for i, sample in enumerate(samples):
            features[i] = self.featurize_sequence(sample.sequence)

        return features

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []

        if self.use_aa_composition:
            names.extend([f'aa_{aa}' for aa in STANDARD_AA])

        if self.use_property_composition:
            names.extend([f'prop_{prop}' for prop in AA_PROPERTIES.keys()])

        if self.use_basic_features:
            names.extend(['length', 'mw', 'charge', 'hydro', 'aromatic'])

        if self.use_dipeptide:
            for aa1 in STANDARD_AA:
                for aa2 in STANDARD_AA:
                    names.append(f'di_{aa1}{aa2}')

        return names
