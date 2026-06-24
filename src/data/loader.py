"""
Data loader for PEM datasets.

Loads train/val/test splits for baselines and PEM training.
"""

from pathlib import Path
from typing import Dict, List, Any
import sys
import torch
from torch.utils.data import Dataset

from .pem_schema import PEMSample
from .serialization import load_samples


def load_pem_dataset(
    dataset: str,
    split: str,
    data_dir: Path = None,
) -> Dict[str, List[PEMSample]]:
    """
    Load PEM dataset with train/val/test splits.

    Args:
        dataset: Dataset name (e.g., "CycPeptMPDB_PAMPA")
        split: Split name (e.g., "scaffold_aware", "random")
        data_dir: Base data directory (default: auto-detect)

    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    if data_dir is None:
        # Default to project data directory
        project_root = Path(__file__).parent.parent.parent

        # Try multiple locations for backward compatibility
        candidate_dirs = [
            project_root / "data" / "processed" / dataset / "splits" / split,
            project_root / "data" / "splits" / dataset / split,
        ]

        split_dir = None
        for candidate in candidate_dirs:
            if candidate.exists():
                split_dir = candidate
                break

        if split_dir is None:
            raise FileNotFoundError(
                f"Split directory not found for dataset={dataset}, split={split}.\n"
                f"Tried:\n" + "\n".join(f"  - {c}" for c in candidate_dirs)
            )
    else:
        split_dir = Path(data_dir) / dataset / "splits" / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Load splits - try both parquet and jsonl formats
    result = {}
    for split_name in ['train', 'val', 'test']:
        # Try parquet first, then jsonl
        parquet_file = split_dir / f"{split_name}.parquet"
        jsonl_file = split_dir / f"{split_name}.jsonl"

        if parquet_file.exists():
            split_file = parquet_file
        elif jsonl_file.exists():
            split_file = jsonl_file
        else:
            raise FileNotFoundError(
                f"Split file not found: tried {parquet_file} and {jsonl_file}"
            )

        samples = load_samples(split_file)
        result[split_name] = samples

    return result


class PEMDataset(Dataset):
    """PyTorch Dataset for PEM training/evaluation."""

    def __init__(self, samples: List[PEMSample]):
        """
        Args:
            samples: List of PEMSample objects
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract required fields
        sequence = sample.sequence
        label = float(sample.label)

        # Extract anchor positions from edits
        anchor_positions = []
        edit_reprs = []

        for edit in sample.edits:
            if edit.anchor_kind == "explicit" and edit.anchor_positions:
                # Take first anchor position for this edit
                anchor_positions.append(edit.anchor_positions[0])
                # Use canonical chemical representation
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
    Collate function for PEM batches.

    Handles variable-length sequences and edits.
    """
    sequences = [item["sequence"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    # Anchor positions stay as lists (variable length per sample)
    anchor_positions = [item["anchor_positions"] for item in batch]
    edit_reprs = [item["edit_reprs"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]

    # For edit_inputs, we'll create a simple representation
    # Each edit is represented by its string for now (descriptor encoder will handle it)
    max_edits = max(len(edits) for edits in edit_reprs) if edit_reprs else 0

    # Placeholder tensor - actual encoding happens in edit encoder
    # For descriptor-based encoder, this will be processed differently
    edit_inputs = edit_reprs  # Keep as list of lists of strings

    return {
        "sequences": sequences,
        "labels": labels,
        "anchor_positions": anchor_positions,
        "edit_inputs": edit_inputs,
        "sample_ids": sample_ids,
    }
