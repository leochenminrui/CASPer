"""
Serialization utilities for PEM schema.

Supports JSONL and Parquet formats for PEMSample data.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from .pem_schema import PEMSample, sample_to_dict


def save_jsonl(samples: List[PEMSample], output_path: Path) -> int:
    """
    Save samples to JSONL format.

    Args:
        samples: List of PEMSample objects
        output_path: Output file path (.jsonl)

    Returns:
        Number of samples written
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in samples:
            sample_dict = sample_to_dict(sample)
            f.write(json.dumps(sample_dict, default=str) + '\n')

    return len(samples)


def load_jsonl(input_path: Path) -> List[PEMSample]:
    """
    Load samples from JSONL format.

    Args:
        input_path: Input file path (.jsonl)

    Returns:
        List of PEMSample objects
    """
    samples = []

    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                sample_dict = json.loads(line)
                sample = PEMSample(**sample_dict)
                samples.append(sample)

    return samples


def save_parquet(samples: List[PEMSample], output_path: Path) -> int:
    """
    Save samples to Parquet format.

    Args:
        samples: List of PEMSample objects
        output_path: Output file path (.parquet)

    Returns:
        Number of samples written

    Note:
        Nested structures (edits, metadata) are serialized as JSON strings
        in the Parquet file for compatibility.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    rows = []
    for sample in samples:
        sample_dict = sample_to_dict(sample)

        # Serialize nested structures as JSON strings
        row = {
            'sample_id': sample_dict['sample_id'],
            'dataset': sample_dict['dataset'],
            'sequence': sample_dict['sequence'],
            'label': sample_dict['label'],
            'label_type': sample_dict['label_type'],
            'label_unit': sample_dict['label_unit'],
            'assay_type': sample_dict['assay_type'],
            'assay_metadata_json': json.dumps(sample_dict['assay_metadata']),
            'edits_json': json.dumps(sample_dict['edits']),
            'anchor_status': sample_dict['anchor_status'],
            'provenance_json': json.dumps(sample_dict['provenance']),
            'split_metadata_json': json.dumps(sample_dict.get('split_metadata')),
            'quality_flags_json': json.dumps(sample_dict['quality_flags'])
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save to Parquet
    df.to_parquet(output_path, index=False, engine='pyarrow')

    return len(samples)


def load_parquet(input_path: Path) -> List[PEMSample]:
    """
    Load samples from Parquet format.

    Args:
        input_path: Input file path (.parquet)

    Returns:
        List of PEMSample objects
    """
    df = pd.read_parquet(input_path, engine='pyarrow')

    samples = []
    for _, row in df.iterrows():
        # Deserialize JSON strings
        sample_dict = {
            'sample_id': row['sample_id'],
            'dataset': row['dataset'],
            'sequence': row['sequence'],
            'label': float(row['label']),
            'label_type': row['label_type'],
            'label_unit': row['label_unit'],
            'assay_type': row['assay_type'],
            'assay_metadata': json.loads(row['assay_metadata_json']),
            'edits': json.loads(row['edits_json']),
            'anchor_status': row['anchor_status'],
            'provenance': json.loads(row['provenance_json']),
            'split_metadata': json.loads(row['split_metadata_json']) if pd.notna(row['split_metadata_json']) else None,
            'quality_flags': json.loads(row['quality_flags_json'])
        }

        sample = PEMSample(**sample_dict)
        samples.append(sample)

    return samples


def save_both_formats(
    samples: List[PEMSample],
    output_dir: Path,
    dataset_name: str
) -> Dict[str, int]:
    """
    Save samples in both JSONL and Parquet formats.

    Args:
        samples: List of PEMSample objects
        output_dir: Output directory
        dataset_name: Dataset name for file naming

    Returns:
        Dict with counts for each format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSONL
    jsonl_path = output_dir / f"{dataset_name}.jsonl"
    jsonl_count = save_jsonl(samples, jsonl_path)

    # Save Parquet
    parquet_path = output_dir / f"{dataset_name}.parquet"
    parquet_count = save_parquet(samples, parquet_path)

    return {
        'jsonl': jsonl_count,
        'parquet': parquet_count,
        'jsonl_path': str(jsonl_path),
        'parquet_path': str(parquet_path)
    }


def load_samples(input_path: Path) -> List[PEMSample]:
    """
    Load samples from either JSONL or Parquet format (auto-detect).

    Args:
        input_path: Input file path (.jsonl or .parquet)

    Returns:
        List of PEMSample objects
    """
    input_path = Path(input_path)

    if input_path.suffix == '.jsonl':
        return load_jsonl(input_path)
    elif input_path.suffix == '.parquet':
        return load_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .jsonl or .parquet")


# Convenience aliases
save_samples_jsonl = save_jsonl
load_samples_jsonl = load_jsonl
save_samples_parquet = save_parquet
load_samples_parquet = load_parquet
