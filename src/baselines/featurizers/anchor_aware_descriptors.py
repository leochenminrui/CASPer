"""
Anchor-Aware Descriptor Featurizer.

Implements the anchor-aware descriptor feature schema for XGBoost baseline.
Version: 1.0.0
"""

from typing import List, Dict, Any, Optional, Set
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from collections import Counter
import warnings

from data.pem_schema import PEMSample


# Amino acid property groups
HYDROPHOBIC_AA = set("ILVAFWM")
CHARGED_AA = set("DEKRH")
POLAR_AA = set("STNQYC")
AROMATIC_AA = set("FYW")
SMALL_AA = set("GAS")


class AnchorAwareDescriptorFeaturizer:
    """
    Anchor-aware descriptor featurizer.

    Combines:
    - Global chemistry features (whole-molecule descriptors)
    - Anchor-aware local features (position, residue, context)
    - Attachment-aware multi-edit features (edit types, structure)
    """

    def __init__(self, descriptor_set: str = "basic", ablation_mode: str = "full"):
        """
        Initialize anchor-aware descriptor featurizer.

        Args:
            descriptor_set: Which descriptor set to use (basic or extended)
            ablation_mode: Feature groups to include:
                - "full": All features (A + B + C)
                - "chemistry_only": Global chemistry only (A)
                - "chemistry_position": A + B1 (position stats)
                - "chemistry_residue": A + B2 (residue composition)
                - "chemistry_context": A + B3 (local context)
                - "chemistry_attachment": A + C (attachment features)
                - "chemistry_anchors": A + B (all anchor features)
        """
        self.descriptor_set = descriptor_set
        self.ablation_mode = ablation_mode

        # Define descriptor functions for global chemistry
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

    def _extract_global_chemistry_features(self, sample: PEMSample) -> List[float]:
        """Extract global chemistry features from SMILES."""
        features = []

        # Get SMILES
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
                        except Exception:
                            features.append(0.0)
                else:
                    # Invalid SMILES
                    features.extend([0.0] * len(self.descriptor_funcs))
            except Exception:
                # Parsing error
                features.extend([0.0] * len(self.descriptor_funcs))
        else:
            # No SMILES
            features.extend([0.0] * len(self.descriptor_funcs))

        # Add edit counts
        features.append(float(len(sample.edits)))
        features.append(float(len(set(e.edit_family for e in sample.edits))))

        return features

    def _extract_anchor_positions_and_residues(self, sample: PEMSample) -> tuple:
        """Extract all anchor positions and residues from edits."""
        all_positions = []
        all_residues = []

        for edit in sample.edits:
            # Skip global edits
            if edit.anchor_kind == "global" or not edit.anchor_positions:
                continue

            # Add positions and residues
            all_positions.extend(edit.anchor_positions)
            all_residues.extend(edit.anchor_residues)

        return all_positions, all_residues

    def _extract_anchor_aware_local_features(self, sample: PEMSample) -> List[float]:
        """Extract anchor-aware local features."""
        features = []

        # Get anchor positions and residues
        all_positions, all_residues = self._extract_anchor_positions_and_residues(sample)

        # B1. Anchor Position Statistics (6 features)
        b1_features = []
        if all_positions:
            anchor_count_total = len(all_positions)
            anchor_count_unique = len(set(all_positions))
            sequence_length = len(sample.sequence)

            b1_features.append(float(anchor_count_total))
            b1_features.append(float(anchor_count_unique))
            b1_features.append(float(anchor_count_unique) / max(sequence_length, 1))
            b1_features.append(float(np.mean(all_positions)))
            b1_features.append(float(np.std(all_positions)) if len(all_positions) > 1 else 0.0)
            b1_features.append(float(max(all_positions) - min(all_positions)) if len(all_positions) > 1 else 0.0)
        else:
            b1_features.extend([0.0] * 6)

        # B2. Anchor Residue Composition (20 features)
        # One-hot encoding for each standard amino acid
        standard_aa = "ACDEFGHIKLMNPQRSTVWY"
        b2_features = []
        if all_residues:
            residue_counts = Counter(all_residues)
            total_residues = len(all_residues)
            for aa in standard_aa:
                b2_features.append(float(residue_counts.get(aa, 0)) / total_residues)
        else:
            b2_features.extend([0.0] * 20)

        # B3. Anchor Local Context (9 features)
        b3_features = []
        if all_residues:
            total = len(all_residues)
            residue_set = set(all_residues)

            hydrophobic_frac = sum(1 for r in all_residues if r in HYDROPHOBIC_AA) / total
            charged_frac = sum(1 for r in all_residues if r in CHARGED_AA) / total
            polar_frac = sum(1 for r in all_residues if r in POLAR_AA) / total
            aromatic_frac = sum(1 for r in all_residues if r in AROMATIC_AA) / total
            small_frac = sum(1 for r in all_residues if r in SMALL_AA) / total
            proline_frac = sum(1 for r in all_residues if r == 'P') / total

            terminal_count = sum(1 for e in sample.edits if e.edit_family in ["n_terminal", "c_terminal"])
            n_terminal_count = sum(1 for e in sample.edits if e.edit_family == "n_terminal")
            c_terminal_count = sum(1 for e in sample.edits if e.edit_family == "c_terminal")

            b3_features.extend([
                hydrophobic_frac,
                charged_frac,
                polar_frac,
                aromatic_frac,
                small_frac,
                proline_frac,
                float(terminal_count) / max(len(sample.edits), 1),
                float(n_terminal_count),
                float(c_terminal_count),
            ])
        else:
            b3_features.extend([0.0] * 9)

        # Store components for ablation
        self._last_b1 = b1_features
        self._last_b2 = b2_features
        self._last_b3 = b3_features

        # Combine all B features
        features.extend(b1_features)
        features.extend(b2_features)
        features.extend(b3_features)

        return features

    def _extract_attachment_aware_features(self, sample: PEMSample) -> List[float]:
        """Extract attachment-aware multi-edit features."""
        features = []

        # C1. Edit Family Distribution (7 features)
        edit_families = [e.edit_family for e in sample.edits]
        family_counts = Counter(edit_families)

        for family in ["n_terminal", "c_terminal", "sidechain", "backbone", "cyclization", "substitution", "other"]:
            features.append(float(family_counts.get(family, 0)))

        # C2. Edit Type Diversity (5 features)
        edit_types = [e.edit_type for e in sample.edits]
        attachment_types = [e.attachment_semantics for e in sample.edits]

        edit_type_count_unique = len(set(edit_types)) if edit_types else 0
        features.append(float(edit_type_count_unique))

        # Shannon entropy for edit types
        if edit_types:
            type_counts = Counter(edit_types)
            total = len(edit_types)
            entropy = -sum((count / total) * np.log2(count / total) for count in type_counts.values())
            features.append(entropy)
        else:
            features.append(0.0)

        attachment_type_count_unique = len(set(attachment_types)) if attachment_types else 0
        features.append(float(attachment_type_count_unique))

        has_cyclization = float(any(e.edit_family == "cyclization" for e in sample.edits))
        has_backbone_mod = float(any(e.edit_family == "backbone" for e in sample.edits))
        features.extend([has_cyclization, has_backbone_mod])

        # C3. Specific Edit Type Counts (10 features)
        edit_type_counts = Counter(edit_types)

        specific_types = [
            "n_methylation",
            "d_amino_acid",
            "non_standard_aa",
            "head_to_tail_cyclization",
            "sidechain_cyclization",
            "disulfide",
            "acetylation",
            "amidation",
            "phosphorylation",
        ]

        for edit_type in specific_types:
            features.append(float(edit_type_counts.get(edit_type, 0)))

        # Other types
        other_count = sum(count for edit_type, count in edit_type_counts.items() if edit_type not in specific_types)
        features.append(float(other_count))

        # C4. Multi-Edit Aggregation (6 features)
        anchor_pair_count = sum(1 for e in sample.edits if len(e.anchor_positions) > 1)
        features.append(float(anchor_pair_count))

        # Extract cyclization ring size if available
        cyclization_ring_size = 0.0
        for e in sample.edits:
            if e.edit_family == "cyclization" and e.edit_metadata:
                ring_size = e.edit_metadata.get("ring_size")
                if ring_size:
                    cyclization_ring_size = float(ring_size)
                    break
        features.append(cyclization_ring_size)

        # Anchor overlap
        all_positions, _ = self._extract_anchor_positions_and_residues(sample)
        position_counts = Counter(all_positions)
        edit_anchor_overlap = sum(1 for count in position_counts.values() if count > 1)
        features.append(float(edit_anchor_overlap))

        # Sequence-level features
        sequence_length = len(sample.sequence)
        sequence_cyclic = float(any(e.edit_family == "cyclization" for e in sample.edits))
        modification_rate = len(sample.edits) / max(sequence_length, 1)

        features.extend([
            float(sequence_length),
            sequence_cyclic,
            modification_rate,
        ])

        return features

    def featurize_sample(self, sample: PEMSample) -> np.ndarray:
        """
        Featurize a single sample.

        Args:
            sample: PEMSample with edits and SMILES

        Returns:
            Feature vector (size depends on ablation_mode)
        """
        features = []

        # A. Global Chemistry Features (10 features)
        a_features = self._extract_global_chemistry_features(sample)
        features.extend(a_features)

        # B. Anchor-Aware Local Features (35 features)
        # Extract all B features to populate self._last_b1, b2, b3
        self._extract_anchor_aware_local_features(sample)

        # C. Attachment-Aware Multi-Edit Features (28 features)
        c_features = self._extract_attachment_aware_features(sample)

        # Apply ablation mode
        if self.ablation_mode == "full":
            # All features: A + B + C
            features.extend(self._last_b1)
            features.extend(self._last_b2)
            features.extend(self._last_b3)
            features.extend(c_features)
        elif self.ablation_mode == "chemistry_only":
            # Only A
            pass  # Already added
        elif self.ablation_mode == "chemistry_position":
            # A + B1
            features.extend(self._last_b1)
        elif self.ablation_mode == "chemistry_residue":
            # A + B2
            features.extend(self._last_b2)
        elif self.ablation_mode == "chemistry_context":
            # A + B3
            features.extend(self._last_b3)
        elif self.ablation_mode == "chemistry_attachment":
            # A + C
            features.extend(c_features)
        elif self.ablation_mode == "chemistry_anchors":
            # A + B (all anchor features)
            features.extend(self._last_b1)
            features.extend(self._last_b2)
            features.extend(self._last_b3)
        elif self.ablation_mode == "site_context_only":
            # B + C only (no chemistry A), for BC model
            # Strip the A features already added at start
            features = []  # clear A features
            features.extend(self._last_b1)
            features.extend(self._last_b2)
            features.extend(self._last_b3)
            features.extend(c_features)
        else:
            raise ValueError(f"Unknown ablation mode: {self.ablation_mode}")

        return np.array(features, dtype=np.float32)

    def featurize(self, samples: List[PEMSample]) -> np.ndarray:
        """
        Featurize multiple samples.

        Args:
            samples: List of PEMSample objects

        Returns:
            Feature matrix (n_samples, 73)
        """
        feature_vectors = []
        for sample in samples:
            feat = self.featurize_sample(sample)
            feature_vectors.append(feat)

        return np.array(feature_vectors)

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []

        # A. Global Chemistry Features (10)
        names.extend(list(self.descriptor_funcs.keys()))
        names.extend(['num_edits', 'num_edit_families'])

        # B. Anchor-Aware Local Features (35)
        # B1. Position stats (6)
        names.extend([
            'anchor_count_total',
            'anchor_count_unique',
            'anchor_density',
            'anchor_pos_mean',
            'anchor_pos_std',
            'anchor_pos_range',
        ])

        # B2. Residue composition (20)
        standard_aa = "ACDEFGHIKLMNPQRSTVWY"
        for aa in standard_aa:
            names.append(f'anchor_res_{aa}')

        # B3. Local context (9)
        names.extend([
            'anchor_hydrophobic_frac',
            'anchor_charged_frac',
            'anchor_polar_frac',
            'anchor_aromatic_frac',
            'anchor_small_frac',
            'anchor_proline_frac',
            'anchor_terminal_frac',
            'anchor_n_terminal_count',
            'anchor_c_terminal_count',
        ])

        # C. Attachment-Aware Multi-Edit Features (28)
        # C1. Edit family distribution (7)
        names.extend([
            'edit_family_n_terminal',
            'edit_family_c_terminal',
            'edit_family_sidechain',
            'edit_family_backbone',
            'edit_family_cyclization',
            'edit_family_substitution',
            'edit_family_other',
        ])

        # C2. Edit type diversity (5)
        names.extend([
            'edit_type_count_unique',
            'edit_type_entropy',
            'attachment_type_count_unique',
            'has_cyclization',
            'has_backbone_mod',
        ])

        # C3. Specific edit type counts (10)
        names.extend([
            'edit_n_methylation',
            'edit_d_amino_acid',
            'edit_non_standard_aa',
            'edit_head_to_tail_cyclization',
            'edit_sidechain_cyclization',
            'edit_disulfide',
            'edit_acetylation',
            'edit_amidation',
            'edit_phosphorylation',
            'edit_other_types',
        ])

        # C4. Multi-edit aggregation (6)
        names.extend([
            'anchor_pair_count',
            'cyclization_ring_size',
            'edit_anchor_overlap',
            'sequence_length',
            'sequence_cyclic',
            'modification_rate',
        ])

        return names
