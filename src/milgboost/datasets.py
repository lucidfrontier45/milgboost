import numpy as np
from sklearn.datasets import make_classification

from .types import LabeledBag


def make_mil_data(
    n_bags: int = 50,
    n_features: int = 10,
    n_instances: tuple[int, int] = (3, 10),
    key_instance_ratio: float = 0.3,
    n_positive_bags: int | None = None,
    noise: float = 0.0,
    informative_ratio: float = 1.0,
    redundant_ratio: float = 0.0,
    n_clusters_per_class: int = 1,
    class_sep: float = 1.0,
    random_state: int | None = None,
) -> list[LabeledBag]:
    rng = np.random.default_rng(random_state)

    bag_sizes = rng.integers(n_instances[0], n_instances[1] + 1, size=n_bags)
    total = int(bag_sizes.sum())

    x, y_inst = make_classification(
        n_samples=total,
        n_features=n_features,
        n_informative=max(1, int(n_features * informative_ratio)),
        n_redundant=int(n_features * redundant_ratio),
        n_repeated=0,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        weights=[0.5, 0.5],
        random_state=random_state,
    )

    perm = rng.permutation(total)
    x = x[perm]
    y_inst = y_inst[perm]

    y = np.empty(n_bags, dtype=np.intp)
    thresholds = np.empty(n_bags, dtype=np.intp)

    boundaries = np.empty(n_bags + 1, dtype=np.intp)
    boundaries[0] = 0
    np.cumsum(bag_sizes, out=boundaries[1:])

    for i in range(n_bags):
        start = boundaries[i]
        end = boundaries[i + 1]
        pos_count = int(y_inst[start:end].sum())
        thresholds[i] = rng.poisson(bag_sizes[i] * key_instance_ratio)
        y[i] = 1 if pos_count >= thresholds[i] else 0

    if n_positive_bags is not None:
        current = int(y.sum())
        if current < n_positive_bags:
            deficits = []
            for i in range(n_bags):
                if y[i] == 0:
                    start = boundaries[i]
                    end = boundaries[i + 1]
                    pos_count = int(y_inst[start:end].sum())
                    deficits.append((thresholds[i] - pos_count, i))
            deficits.sort()
            for _, idx in deficits[: n_positive_bags - current]:
                y[idx] = 1
        elif current > n_positive_bags:
            surpluses = []
            for i in range(n_bags):
                if y[i] == 1:
                    start = boundaries[i]
                    end = boundaries[i + 1]
                    pos_count = int(y_inst[start:end].sum())
                    surpluses.append((pos_count - thresholds[i], i))
            surpluses.sort()
            for _, idx in surpluses[: current - n_positive_bags]:
                y[idx] = 0

    if noise > 0.0:
        n_flip = int(noise * n_bags)
        if n_flip > 0:
            flip_idx = rng.choice(n_bags, size=n_flip, replace=False)
            y[flip_idx] = 1 - y[flip_idx]

    bags = []
    for i in range(n_bags):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        bags.append(LabeledBag(features=x[start:end], label=int(y[i])))
    return bags
