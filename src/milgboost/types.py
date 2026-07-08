from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Bag:
    features: np.ndarray


@dataclass(slots=True)
class LabeledBag(Bag):
    label: int


def validate_dense_bag_ids(z: np.ndarray) -> int:
    """Validate that ``z`` is dense 0-based contiguous bag IDs.

    MIL paths index bag-level arrays by the raw ``z`` value (``LSEBCE``,
    ``arrays_to_labeled_bags``, model ``fit``). Gaps or a non-zero minimum
    would yield wrong labels, ``-inf`` logits, or index errors.

    Args:
        z: Per-instance bag IDs.

    Returns:
        Number of distinct bags (``z.max() + 1``).

    Raises:
        ValueError: If ``z`` is empty, not 0-based, or has gaps.
    """
    arr = np.asarray(z)
    if arr.size == 0:
        raise ValueError("bag_ids z must be non-empty")
    z_min = int(arr.min())
    z_max = int(arr.max())
    n_unique = int(np.unique(arr).size)
    if z_min != 0 or n_unique != z_max + 1:
        raise ValueError(
            f"bag_ids z must be dense 0-based contiguous "
            f"(got min={z_min}, max={z_max}, unique={n_unique}); "
            f"remap via np.unique(z, return_inverse=True)"
        )
    return z_max + 1


def arrays_to_bags(x: np.ndarray, z: np.ndarray) -> list[Bag]:
    unique_z = sorted(np.unique(z))
    return [Bag(features=x[z == i]) for i in unique_z]


def bags_to_arrays(bags: Sequence[Bag]) -> tuple[np.ndarray, np.ndarray]:
    if not bags:
        return np.empty((0, 0)), np.empty(0, dtype=np.intp)
    features = np.vstack([b.features for b in bags])
    bag_sizes = [b.features.shape[0] for b in bags]
    z = np.repeat(np.arange(len(bags), dtype=np.intp), bag_sizes)
    return features, z


def arrays_to_labeled_bags(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> list[LabeledBag]:
    n_bags = validate_dense_bag_ids(z)
    if len(y) != n_bags:
        raise ValueError(f"y length {len(y)} != number of bags {n_bags} implied by z")
    unique_z = sorted(np.unique(z))
    return [LabeledBag(features=x[z == i], label=int(y[i])) for i in unique_z]


def labeled_bags_to_arrays(
    bags: Sequence[LabeledBag],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not bags:
        return np.empty((0, 0)), np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    features = np.vstack([b.features for b in bags])
    bag_sizes = [b.features.shape[0] for b in bags]
    z = np.repeat(np.arange(len(bags), dtype=np.intp), bag_sizes)
    y = np.array([b.label for b in bags], dtype=np.intp)
    return features, y, z
