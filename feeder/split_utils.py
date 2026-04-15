"""
Subject-independent (or session-independent) train/val/test indexing.

Replaces window-level random shuffle + ratio split: first shuffle *subject* (or session) IDs
with a fixed seed, assign subjects to splits by train_split/val_split/test_split counts,
then keep all windows from subjects in the requested split.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def subject_independent_indices(
    sample_units: List[Any],
    split_name: str,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Args:
        sample_units: Per-sample unit id (subject id or session id), same length as windows.
        split_name: 'train' | 'val' | 'test'
    Returns:
        indices: indices into the parallel samples list for this split.
        info: counts and which units went to which split (for logging).
    """
    import random

    assert split_name in ("train", "val", "test")
    assert abs(train_split + val_split + test_split - 1.0) < 1e-5

    n_samples = len(sample_units)
    if n_samples == 0:
        return [], {
            "n_units": 0,
            "train_units": [],
            "val_units": [],
            "test_units": [],
        }

    unique = sorted(set(sample_units), key=lambda x: (str(type(x)), str(x)))
    rng = random.Random(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    n = len(shuffled)

    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val

    # With few subjects, int(n*val_split) may be 0 and val/test become empty.
    # Fallback: steal from test first to keep train size when possible.
    if n == 2:
        # At least ensure train and val are non-empty
        n_train, n_val, n_test = 1, 1, 0
    elif n >= 3:
        if n_val == 0:
            n_val = 1
            if n_test > 1:
                n_test -= 1
            elif n_train > 1:
                n_train -= 1
            else:
                n_test = max(0, n - n_train - n_val)
        if n_test == 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_test = max(0, n - n_train - n_val)

    assert n_train + n_val + n_test == n, (n, n_train, n_val, n_test)

    train_units = set(shuffled[:n_train])
    val_units = set(shuffled[n_train : n_train + n_val])
    test_units = set(shuffled[n_train + n_val :])

    if split_name == "train":
        target = train_units
    elif split_name == "val":
        target = val_units
    else:
        target = test_units

    indices = [i for i, u in enumerate(sample_units) if u in target]

    info = {
        "n_units": n,
        "n_train_units": n_train,
        "n_val_units": n_val,
        "n_test_units": n_test,
        "train_units": sorted(train_units, key=str),
        "val_units": sorted(val_units, key=str),
        "test_units": sorted(test_units, key=str),
    }
    return indices, info
