"""
Shared train/validation/test split logic for traditional ML and deep learning workflows.

Produces a stratified 75:5:20 (train:val:test) split on environment. The split is defined
on SRA samples (sample_id matching SRR*, ERR*, or DRR*) from sample_data after filtering
for Bacteria domain and non-missing pH. Both workflows use the same random seed and
identical test set; traditional ML uses 80% for training (train+val), deep learning
uses 75% train and 5% validation.
"""

import re

import pandas as pd
from sklearn.model_selection import train_test_split

# Default random seed; same for both workflows
DEFAULT_RANDOM_SEED = 42

# Split proportions: 75% train, 5% val, 20% test
TEST_SIZE = 0.2  # 20% test

# SRA sample_id pattern (SRR*, ERR*, DRR*)
SRA_PATTERN = re.compile(r"^(SRR|ERR|DRR)\d+", re.IGNORECASE)


def is_sra_sample(sample_id: str) -> bool:
    """Return True if sample_id is an SRA identifier (SRR*, ERR*, DRR*)."""
    return bool(sample_id and SRA_PATTERN.match(str(sample_id)))


def get_train_val_test_ids(
    sample_data_path: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
):
    """
    Compute stratified train/val/test split on SRA samples (SRR*, ERR*, DRR*).

    Uses sample_data after filtering for Bacteria domain, non-missing pH, and SRA samples.
    Stratifies on environment. Proportions: 75% train, 5% val, 20% test (same random seed
    for both workflows).

    Parameters
    ----------
    sample_data_path : str
        Path to sample_data.csv
    random_seed : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    tuple of (set, set, set)
        (train_sample_ids, val_sample_ids, test_sample_ids), each a set of sample_id strings.
        Only includes SRA samples after filtering.
    """
    sample_data = pd.read_csv(sample_data_path)
    sample_data = sample_data[sample_data["domain"] == "Bacteria"].copy()
    sample_data = sample_data.dropna(subset=["pH"])

    # Restrict to SRA samples
    sra_mask = sample_data["sample_id"].astype(str).apply(is_sra_sample)
    sample_data = sample_data.loc[sra_mask].reset_index(drop=True)

    sample_ids = sample_data["sample_id"].values
    environment = sample_data["environment"].values

    # 80% train+val, 20% test (stratified on environment)
    ids_trainval, ids_test, env_trainval, _ = train_test_split(
        sample_ids.tolist(),
        environment.tolist(),
        test_size=TEST_SIZE,
        random_state=random_seed,
        stratify=environment,
    )

    # Subdivide train+val: val = 5% of total, train = 75% (stratified on environment)
    n_trainval = len(ids_trainval)
    n_val = int(round(len(sample_ids) * 0.05))
    val_size = n_val / n_trainval

    ids_train, ids_val = train_test_split(
        ids_trainval,
        test_size=val_size,
        random_state=random_seed,
        stratify=env_trainval,
    )

    return (
        set(ids_train),
        set(ids_val),
        set(ids_test),
    )
