"""
Chemical descriptor featurizers.

Extracts molecular descriptors from chemical modifications for baselines.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from data.pem_schema import PEMSample


class ChemicalDescriptorFeaturizer:
    """
    Extract chemical descriptors from modifications.

    This featurizer computes molecular descriptors for each chemical modification
    and aggregates them (no positional information used).
    """

    def __init__(
        self,
        aggregation: str = "mean",  # mean, sum, concat, max
        descriptor_set: str = "basic",  # basic, extended, all
        use_counts: bool = True,  # Include edit count features
    ):
        """
        Initialize descriptor featurizer.

        Args:
            aggregation: How to aggregate descriptors across modifications
            descriptor_set: Which descriptor set to use
            use_counts: Whether to include modification count features
        """
        self.aggregation = aggregation
        self.descriptor_set = descriptor_set
        self.use_counts = use_counts

        # Define descriptor functions
        if descriptor_set == "basic":
            self.descriptor_funcs = {
                'mol_weight': Descriptors.MolWt,
                'logp': Descriptors.MolLogP,
                'tpsa': Descriptors.TPSA,
                'num_h_acceptors': Descriptors.NumHAcceptors,
                'num_h_donors': Descriptors.NumHDonors,
                'num_rotatable_bonds': Descriptors.NumRotatableBonds,
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings,
                'num_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings,
            }
        elif descriptor_set == "extended":
            self.descriptor_funcs = {
                'mol_weight': Descriptors.MolWt,
                'logp': Descriptors.MolLogP,
                'tpsa': Descriptors.TPSA,
                'num_h_acceptors': Descriptors.NumHAcceptors,
                'num_h_donors': Descriptors.NumHDonors,
                'num_rotatable_bonds': Descriptors.NumRotatableBonds,
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings,
                'num_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings,
                'num_heteroatoms': Descriptors.NumHeteroatoms,
                'num_valence_electrons': Descriptors.NumValenceElectrons,
                'molar_refractivity': Descriptors.MolMR,
                'fraction_csp3': rdMolDescriptors.CalcFractionCsp3,
                'chi0n': rdMolDescriptors.CalcChi0n,
                'chi1n': rdMolDescriptors.CalcChi1n,
                'kappa1': rdMolDescriptors.CalcKappa1,
            }
        else:  # all
            # Get all available descriptors (warning: this is >200 descriptors)
            self.descriptor_funcs = {
                name: getattr(Descriptors, name)
                for name in dir(Descriptors)
                if not name.startswith('_') and callable(getattr(Descriptors, name))
            }

    def featurize_molecule(self, smiles: str) -> Optional[np.ndarray]:
        """
        Compute descriptors for a molecule from SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Descriptor vector or None if SMILES invalid
        """
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Compute descriptors
            descriptors = []
            for func in self.descriptor_funcs.values():
                try:
                    value = func(mol)
                    descriptors.append(float(value) if value is not None else 0.0)
                except:
                    descriptors.append(0.0)

            return np.array(descriptors)

        except Exception:
            return None

    def featurize_sample(self, sample: PEMSample) -> np.ndarray:
        """
        Featurize a single sample using aggregated edit descriptors.

        Args:
            sample: PEMSample with chemical edits

        Returns:
            Feature vector
        """
        features = []

        # Count features
        if self.use_counts:
            count_features = [
                len(sample.edits),  # Total number of edits
                len(set(e.edit_family for e in sample.edits)),  # Number of unique families
            ]
            features.extend(count_features)

        # NOTE: For now, we can't compute per-edit descriptors since edits don't have SMILES
        # Instead, use edit type counts as features
        edit_family_counts = {}
        for family in ['n_terminal', 'c_terminal', 'sidechain', 'backbone', 'cyclization', 'substitution', 'other']:
            edit_family_counts[family] = sum(1 for e in sample.edits if e.edit_family == family)

        features.extend(edit_family_counts.values())

        # Placeholder descriptors (zeros) since we can't compute per-edit descriptors
        dummy_desc = np.zeros(len(self.descriptor_funcs))
        features.extend(dummy_desc)

        return np.array(features)

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

        if self.use_counts:
            names.extend(['num_edits', 'num_edit_families'])

        # Descriptor names
        desc_names = list(self.descriptor_funcs.keys())

        if self.aggregation == "concat":
            max_mods = 5
            for i in range(max_mods):
                for name in desc_names:
                    names.append(f"mod{i}_{name}")
        else:
            for name in desc_names:
                names.append(f"{self.aggregation}_{name}")

        return names


class DescriptorOnlyFeaturizer:
    """
    Descriptor-only baseline (no sequence information).

    Uses whole-molecule chemical descriptors from SMILES.
    """

    def __init__(
        self,
        aggregation: str = "mean",  # Ignored for molecule-level descriptors
        descriptor_set: str = "basic",
        use_counts: bool = True,
    ):
        """
        Initialize descriptor-only featurizer.

        Args:
            aggregation: Ignored (kept for API compatibility)
            descriptor_set: Which descriptor set to use
            use_counts: Whether to include modification count features
        """
        self.descriptor_set = descriptor_set
        self.use_counts = use_counts

        # Define descriptor functions (same as ChemicalDescriptorFeaturizer)
        if descriptor_set == "basic":
            self.descriptor_funcs = {
                'mol_weight': Descriptors.MolWt,
                'logp': Descriptors.MolLogP,
                'tpsa': Descriptors.TPSA,
                'num_h_acceptors': Descriptors.NumHAcceptors,
                'num_h_donors': Descriptors.NumHDonors,
                'num_rotatable_bonds': Descriptors.NumRotatableBonds,
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings,
                'num_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings,
            }
        elif descriptor_set == "extended":
            self.descriptor_funcs = {
                'mol_weight': Descriptors.MolWt,
                'logp': Descriptors.MolLogP,
                'tpsa': Descriptors.TPSA,
                'num_h_acceptors': Descriptors.NumHAcceptors,
                'num_h_donors': Descriptors.NumHDonors,
                'num_rotatable_bonds': Descriptors.NumRotatableBonds,
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings,
                'num_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings,
                'num_heteroatoms': Descriptors.NumHeteroatoms,
                'num_valence_electrons': Descriptors.NumValenceElectrons,
                'molar_refractivity': Descriptors.MolMR,
                'fraction_csp3': rdMolDescriptors.CalcFractionCsp3,
                'chi0n': rdMolDescriptors.CalcChi0n,
                'chi1n': rdMolDescriptors.CalcChi1n,
                'kappa1': rdMolDescriptors.CalcKappa1,
            }
        else:
            raise ValueError(f"Unknown descriptor set: {descriptor_set}")

    def featurize_sample(self, sample: PEMSample) -> np.ndarray:
        """
        Featurize a single sample using whole-molecule descriptors.

        Args:
            sample: PEMSample with SMILES in assay_metadata

        Returns:
            Feature vector
        """
        features = []

        # Count features
        if self.use_counts:
            count_features = [
                len(sample.edits),  # Total number of edits
                len(set(e.edit_family for e in sample.edits)),  # Number of unique families
            ]
            features.extend(count_features)

        # Get SMILES from assay_metadata
        smiles = sample.assay_metadata.get('smiles')

        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Compute descriptors
                    for func in self.descriptor_funcs.values():
                        try:
                            value = func(mol)
                            features.append(float(value) if value is not None else 0.0)
                        except:
                            features.append(0.0)
                else:
                    # Invalid SMILES - use zeros
                    features.extend([0.0] * len(self.descriptor_funcs))
            except:
                # Parsing error - use zeros
                features.extend([0.0] * len(self.descriptor_funcs))
        else:
            # No SMILES available - use zeros
            features.extend([0.0] * len(self.descriptor_funcs))

        return np.array(features)

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

        if self.use_counts:
            names.extend(['num_edits', 'num_edit_families'])

        # Descriptor names
        names.extend(list(self.descriptor_funcs.keys()))

        return names
