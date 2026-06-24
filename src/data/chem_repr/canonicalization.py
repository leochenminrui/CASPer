"""
Canonicalization rules for chemical edits (v1.0).

Conservative scope:
- ncAA substitution
- Simple cyclization
- Simple sidechain modifications

All rules are versioned and auditable.
"""

from typing import Optional, Tuple
import re

from .representation import (
    ChemRepr,
    Attachment,
    BridgeAttachment,
    SequenceContext,
    ParserStatus,
    ExclusionReason,
    validate_attachment_chemistry
)


class CanonicalizerationRule:
    """Base class for canonicalization rules."""

    def __init__(self, rule_id: str, version: str = "v1"):
        self.rule_id = f"{rule_id}_{version}"
        self.version = version

    def can_parse(self, edit_data: dict) -> bool:
        """Check if this rule can handle the edit."""
        raise NotImplementedError

    def parse(
        self,
        edit_data: dict,
        sequence: str
    ) -> Tuple[Optional[ChemRepr], Optional[str]]:
        """
        Parse edit to ChemRepr.

        Args:
            edit_data: Edit information from PEM schema
            sequence: Full peptide sequence

        Returns:
            (ChemRepr or None, error_message or None)
        """
        raise NotImplementedError


class DAminoAcidRule(CanonicalizerationRule):
    """
    Rule: ncaa_d_amino_acid_v1

    Handles D-amino acid substitutions.
    """

    def __init__(self):
        super().__init__("ncaa_d_amino_acid", "v1")

        # D-amino acid SMILES (inverted chirality)
        self.d_aa_smiles = {
            'A': 'C[C@H](N)C(=O)O',     # D-Ala
            'C': 'C(C[C@H](C(=O)O)N)S', # D-Cys
            'D': 'C([C@H](C(=O)O)N)C(=O)O',  # D-Asp
            'E': 'C(CC(=O)O)[C@H](C(=O)O)N',  # D-Glu
            'F': 'C1=CC=C(C=C1)C[C@H](C(=O)O)N',  # D-Phe
            'G': 'C(C(=O)O)N',          # Gly (achiral)
            'H': 'C1=C(NC=N1)C[C@H](C(=O)O)N',  # D-His
            'I': 'CC[C@@H](C)[C@H](C(=O)O)N',   # D-Ile
            'K': 'C(CCN)C[C@H](C(=O)O)N',  # D-Lys
            'L': 'CC(C)C[C@H](C(=O)O)N',   # D-Leu
            'M': 'CSCC[C@H](C(=O)O)N',     # D-Met
            'N': 'C([C@H](C(=O)O)N)C(=O)N',  # D-Asn
            'P': 'C1C[C@H](NC1)C(=O)O',    # D-Pro
            'Q': 'C(CC(=O)N)[C@H](C(=O)O)N',  # D-Gln
            'R': 'C(C[C@H](C(=O)O)N)CNC(=N)N',  # D-Arg
            'S': 'C([C@H](C(=O)O)N)O',     # D-Ser
            'T': 'C[C@H]([C@H](C(=O)O)N)O',  # D-Thr
            'V': 'CC(C)[C@H](C(=O)O)N',    # D-Val
            'W': 'C1=CC=C2C(=C1)C(=CN2)C[C@H](C(=O)O)N',  # D-Trp
            'Y': 'C1=CC(=CC=C1C[C@H](C(=O)O)N)O'  # D-Tyr
        }

    def can_parse(self, edit_data: dict) -> bool:
        """Check if edit is D-amino acid."""
        return (
            edit_data.get('edit_family') == 'substitution' and
            edit_data.get('edit_type') == 'd_amino_acid'
        )

    def parse(
        self,
        edit_data: dict,
        sequence: str
    ) -> Tuple[Optional[ChemRepr], Optional[str]]:
        """Parse D-amino acid edit."""

        # Extract position and residue
        if not edit_data.get('anchor_positions'):
            return None, "Missing anchor position"

        position = edit_data['anchor_positions'][0]
        if position < 0 or position >= len(sequence):
            return None, f"Invalid position {position}"

        residue = sequence[position]

        # Check if we have SMILES for this residue
        if residue not in self.d_aa_smiles:
            return None, f"No D-amino acid SMILES for {residue}"

        # Create attachment (D-AA replaces entire residue at CA)
        attachment = Attachment(
            position=position,
            residue=residue,
            atom='CA',
            bond='single'
        )

        # Validate
        if not attachment.validate(sequence):
            return None, "Attachment validation failed"

        # Extract context
        context = SequenceContext.from_sequence(sequence, position, window=3)

        # Create representation
        chem_repr = ChemRepr(
            edit_id=edit_data.get('edit_id', f'edit_{position}'),
            edit_family='substitution',
            edit_type='d_amino_acid',
            moiety_smiles=self.d_aa_smiles[residue],
            moiety_name=f"d{residue}",
            attachment=attachment,
            context=context,
            rule_id=self.rule_id,
            parser_status=ParserStatus.SUCCESS,
            canonicalization_notes=f"D-amino acid replacement for {residue} at position {position}",
            input_raw=edit_data.get('chem_rep_raw', f'd{residue}')
        )

        return chem_repr, None


class DisulfideBridgeRule(CanonicalizerationRule):
    """
    Rule: cyclization_disulfide_v1

    Handles Cys-Cys disulfide bridges.
    """

    def __init__(self):
        super().__init__("cyclization_disulfide", "v1")

    def can_parse(self, edit_data: dict) -> bool:
        """Check if edit is disulfide bridge."""
        return (
            edit_data.get('edit_family') == 'cyclization' and
            edit_data.get('edit_type') == 'disulfide_bridge' and
            len(edit_data.get('anchor_positions', [])) == 2
        )

    def parse(
        self,
        edit_data: dict,
        sequence: str
    ) -> Tuple[Optional[ChemRepr], Optional[str]]:
        """Parse disulfide bridge."""

        positions = edit_data['anchor_positions']
        if len(positions) != 2:
            return None, "Disulfide requires exactly 2 positions"

        pos1, pos2 = sorted(positions)

        # Validate positions
        if pos1 < 0 or pos2 >= len(sequence):
            return None, f"Invalid positions {pos1}, {pos2}"

        if sequence[pos1] != 'C' or sequence[pos2] != 'C':
            return None, f"Both positions must be Cys, got {sequence[pos1]}, {sequence[pos2]}"

        # Create attachments
        attach1 = Attachment(
            position=pos1,
            residue='C',
            atom='SG',
            bond='single'
        )

        attach2 = Attachment(
            position=pos2,
            residue='C',
            atom='SG',
            bond='single'
        )

        bridge = BridgeAttachment(attach1, attach2)

        # Extract contexts
        context1 = SequenceContext.from_sequence(sequence, pos1, window=3)
        context2 = SequenceContext.from_sequence(sequence, pos2, window=3)

        # Create representation
        chem_repr = ChemRepr(
            edit_id=edit_data.get('edit_id', f'edit_{pos1}_{pos2}'),
            edit_family='cyclization',
            edit_type='disulfide_bridge',
            moiety_smiles='S-S',  # Disulfide bond
            moiety_name='disulfide',
            attachment=attach1,  # Primary anchor
            context=context1,
            bridge_attachment=bridge,
            bridge_context=context2,
            rule_id=self.rule_id,
            parser_status=ParserStatus.SUCCESS,
            canonicalization_notes=f"Disulfide bridge between Cys{pos1} and Cys{pos2}",
            input_raw=edit_data.get('chem_rep_raw', f'Cys{pos1}-Cys{pos2}')
        )

        return chem_repr, None


class AcetylationRule(CanonicalizerationRule):
    """
    Rule: sidechain_acetylation_v1

    Handles acetylation on Lys or N-terminus.
    """

    def __init__(self):
        super().__init__("sidechain_acetylation", "v1")

    def can_parse(self, edit_data: dict) -> bool:
        """Check if edit is acetylation."""
        return (
            edit_data.get('edit_type') == 'acetylation' and
            edit_data.get('edit_family') in ['sidechain', 'n_terminal']
        )

    def parse(
        self,
        edit_data: dict,
        sequence: str
    ) -> Tuple[Optional[ChemRepr], Optional[str]]:
        """Parse acetylation."""

        if not edit_data.get('anchor_positions'):
            return None, "Missing anchor position"

        position = edit_data['anchor_positions'][0]
        if position < 0 or position >= len(sequence):
            return None, f"Invalid position {position}"

        residue = sequence[position]

        # Determine attachment atom
        if edit_data.get('edit_family') == 'n_terminal' and position == 0:
            atom = 'N'  # N-terminal
        elif residue == 'K':
            atom = 'NZ'  # Lysine sidechain
        else:
            return None, f"Acetylation not supported for {residue} at position {position}"

        # Create attachment
        attachment = Attachment(
            position=position,
            residue=residue,
            atom=atom,
            bond='single'
        )

        # Validate
        if not validate_attachment_chemistry(residue, atom):
            return None, f"Invalid attachment: {residue}.{atom}"

        # Extract context
        context = SequenceContext.from_sequence(sequence, position, window=3)

        # Create representation
        chem_repr = ChemRepr(
            edit_id=edit_data.get('edit_id', f'edit_{position}'),
            edit_family=edit_data['edit_family'],
            edit_type='acetylation',
            moiety_smiles='CC(=O)',  # Acetyl group
            moiety_name='Ac',
            attachment=attachment,
            context=context,
            rule_id=self.rule_id,
            parser_status=ParserStatus.SUCCESS,
            canonicalization_notes=f"Acetylation on {residue}{position} at {atom}",
            input_raw=edit_data.get('chem_rep_raw', f'Ac-{residue}')
        )

        return chem_repr, None


class PhosphorylationRule(CanonicalizerationRule):
    """
    Rule: sidechain_phosphorylation_v1

    Handles phosphorylation on Ser, Thr, Tyr.
    """

    def __init__(self):
        super().__init__("sidechain_phosphorylation", "v1")

        # Phosphorylatable residues and attachment atoms
        self.phospho_targets = {
            'S': 'OG',   # Serine
            'T': 'OG1',  # Threonine
            'Y': 'OH'    # Tyrosine
        }

    def can_parse(self, edit_data: dict) -> bool:
        """Check if edit is phosphorylation."""
        return (
            edit_data.get('edit_family') == 'sidechain' and
            edit_data.get('edit_type') == 'phosphorylation'
        )

    def parse(
        self,
        edit_data: dict,
        sequence: str
    ) -> Tuple[Optional[ChemRepr], Optional[str]]:
        """Parse phosphorylation."""

        if not edit_data.get('anchor_positions'):
            return None, "Missing anchor position"

        position = edit_data['anchor_positions'][0]
        if position < 0 or position >= len(sequence):
            return None, f"Invalid position {position}"

        residue = sequence[position]

        # Check if residue is phosphorylatable
        if residue not in self.phospho_targets:
            return None, f"Phosphorylation not supported for {residue}"

        atom = self.phospho_targets[residue]

        # Create attachment
        attachment = Attachment(
            position=position,
            residue=residue,
            atom=atom,
            bond='single'
        )

        # Validate
        if not validate_attachment_chemistry(residue, atom):
            return None, f"Invalid attachment: {residue}.{atom}"

        # Extract context
        context = SequenceContext.from_sequence(sequence, position, window=3)

        # Create representation
        chem_repr = ChemRepr(
            edit_id=edit_data.get('edit_id', f'edit_{position}'),
            edit_family='sidechain',
            edit_type='phosphorylation',
            moiety_smiles='OP(=O)(O)O',  # Phosphate group
            moiety_name='PO3',
            attachment=attachment,
            context=context,
            rule_id=self.rule_id,
            parser_status=ParserStatus.SUCCESS,
            canonicalization_notes=f"Phosphorylation on {residue}{position} at {atom}",
            input_raw=edit_data.get('chem_rep_raw', f'p{residue}')
        )

        return chem_repr, None


# Registry of all rules
CANONICALIZATION_RULES = [
    DAminoAcidRule(),
    DisulfideBridgeRule(),
    AcetylationRule(),
    PhosphorylationRule()
]


def canonicalize_edit(
    edit_data: dict,
    sequence: str
) -> Tuple[Optional[ChemRepr], Optional[str], str]:
    """
    Canonicalize an edit using appropriate rule.

    Args:
        edit_data: Edit information from PEM schema
        sequence: Full peptide sequence

    Returns:
        (ChemRepr or None, error_message or None, status_notes)
    """
    # Try each rule in order
    for rule in CANONICALIZATION_RULES:
        if rule.can_parse(edit_data):
            chem_repr, error = rule.parse(edit_data, sequence)

            if chem_repr:
                # Success
                return chem_repr, None, f"Parsed with {rule.rule_id}"
            else:
                # Rule matched but failed
                return None, error, f"Failed with {rule.rule_id}: {error}"

    # No rule matched
    return None, "No matching canonicalization rule", "unsupported_modification"
