"""
Mechanism control datasets for PEM experiments.

Implements:
1. Wrong-anchor control: replaces correct anchors with incorrect positions
2. Coarse-position control: replaces exact anchors with coarser positional signals
"""

from typing import List, Dict, Any
import random
import torch
from torch.utils.data import Dataset

from .pem_schema import PEMSample


class WrongAnchorDataset(Dataset):
    """
    Dataset that replaces correct anchor positions with wrong ones.

    Strategy: For each edit, shift the anchor position by a random offset
    that ensures it's different but still within the sequence bounds.
    """

    def __init__(self, samples: List[PEMSample], seed: int = 42):
        """
        Args:
            samples: List of PEMSample objects
            seed: Random seed for reproducible anchor perturbation
        """
        self.samples = samples
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        sequence = sample.sequence
        label = float(sample.label)
        seq_len = len(sequence)

        anchor_positions = []
        edit_reprs = []

        for edit in sample.edits:
            if edit.anchor_kind == "explicit" and edit.anchor_positions:
                # Get original anchor
                original_anchor = edit.anchor_positions[0]

                # Generate wrong anchor by shifting
                # Use deterministic shift based on sample_id and original position
                # to ensure reproducibility
                shift_seed = hash((sample.sample_id, original_anchor)) % 2**31
                local_rng = random.Random(shift_seed)

                # Try to shift by 1-5 positions (or more if needed)
                # Ensure we get a different valid position
                attempts = 0
                max_attempts = 20
                wrong_anchor = original_anchor

                while wrong_anchor == original_anchor and attempts < max_attempts:
                    # Random shift between -5 and +5 (excluding 0)
                    shift = local_rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                    candidate = original_anchor + shift

                    # Keep within bounds
                    if 0 <= candidate < seq_len:
                        wrong_anchor = candidate

                    attempts += 1

                # If we couldn't find a different position, use opposite end of sequence
                if wrong_anchor == original_anchor:
                    if original_anchor < seq_len // 2:
                        wrong_anchor = seq_len - 1
                    else:
                        wrong_anchor = 0

                anchor_positions.append(wrong_anchor)
                edit_reprs.append(edit.chem_rep_canonical)

        return {
            "sequence": sequence,
            "label": label,
            "anchor_positions": anchor_positions,
            "edit_reprs": edit_reprs,
            "sample_id": sample.sample_id,
        }


class CoarsePositionDataset(Dataset):
    """
    Dataset that replaces exact anchor positions with coarse positional signals.

    Strategy: Instead of exact position, use a region indicator (start/middle/end)
    or add noise to the position.
    """

    def __init__(self, samples: List[PEMSample], strategy: str = "tertile", seed: int = 42):
        """
        Args:
            samples: List of PEMSample objects
            strategy: "tertile" (start/middle/end regions) or "noise" (add random noise)
            seed: Random seed for reproducibility
        """
        self.samples = samples
        self.strategy = strategy
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        sequence = sample.sequence
        label = float(sample.label)
        seq_len = len(sequence)

        anchor_positions = []
        edit_reprs = []

        for edit in sample.edits:
            if edit.anchor_kind == "explicit" and edit.anchor_positions:
                original_anchor = edit.anchor_positions[0]

                if self.strategy == "tertile":
                    # Map to tertile center
                    tertile_size = seq_len // 3

                    if original_anchor < tertile_size:
                        # Start region -> center of first tertile
                        coarse_anchor = tertile_size // 2
                    elif original_anchor < 2 * tertile_size:
                        # Middle region -> center of second tertile
                        coarse_anchor = tertile_size + tertile_size // 2
                    else:
                        # End region -> center of third tertile
                        coarse_anchor = 2 * tertile_size + tertile_size // 2

                    # Ensure within bounds
                    coarse_anchor = min(max(0, coarse_anchor), seq_len - 1)

                elif self.strategy == "noise":
                    # Add random noise (±3 positions)
                    noise_seed = hash((sample.sample_id, original_anchor)) % 2**31
                    local_rng = random.Random(noise_seed)

                    noise = local_rng.randint(-3, 3)
                    coarse_anchor = original_anchor + noise

                    # Ensure within bounds
                    coarse_anchor = min(max(0, coarse_anchor), seq_len - 1)

                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")

                anchor_positions.append(coarse_anchor)
                edit_reprs.append(edit.chem_rep_canonical)

        return {
            "sequence": sequence,
            "label": label,
            "anchor_positions": anchor_positions,
            "edit_reprs": edit_reprs,
            "sample_id": sample.sample_id,
        }


class GradedAnchorPerturbationDataset(Dataset):
    """
    Dataset that shifts anchors by a fixed distance.

    This implements graded perturbation to test position-sensitivity.
    Unlike WrongAnchorDataset which randomizes, this shifts by exact offsets.
    """

    def __init__(self, samples: List[PEMSample], shift_distance: int, seed: int = 42):
        """
        Args:
            samples: List of PEMSample objects
            shift_distance: How many residues to shift (+/- value, 0 for no shift)
            seed: Random seed for tie-breaking direction
        """
        self.samples = samples
        self.shift_distance = shift_distance
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        sequence = sample.sequence
        label = float(sample.label)
        seq_len = len(sequence)

        anchor_positions = []
        edit_reprs = []

        for edit in sample.edits:
            if edit.anchor_kind == "explicit" and edit.anchor_positions:
                original_anchor = edit.anchor_positions[0]

                if self.shift_distance == 0:
                    # No shift (baseline)
                    perturbed_anchor = original_anchor
                else:
                    # Apply shift
                    perturbed_anchor = original_anchor + self.shift_distance

                    # Keep within bounds
                    if perturbed_anchor < 0:
                        perturbed_anchor = 0
                    elif perturbed_anchor >= seq_len:
                        perturbed_anchor = seq_len - 1

                anchor_positions.append(perturbed_anchor)
                edit_reprs.append(edit.chem_rep_canonical)

        return {
            "sequence": sequence,
            "label": label,
            "anchor_positions": anchor_positions,
            "edit_reprs": edit_reprs,
            "sample_id": sample.sample_id,
        }


def pem_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for mechanism control batches.
    Same as standard PEM collate.
    """
    sequences = [item["sequence"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    anchor_positions = [item["anchor_positions"] for item in batch]
    edit_reprs = [item["edit_reprs"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]

    edit_inputs = edit_reprs

    return {
        "sequences": sequences,
        "labels": labels,
        "anchor_positions": anchor_positions,
        "edit_inputs": edit_inputs,
        "sample_ids": sample_ids,
    }
