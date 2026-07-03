from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Self

import numpy as np

from ..types import Bag, LabeledBag, bags_to_arrays, labeled_bags_to_arrays


def instance_max_pooling(raw_preds: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Aggregate instance predictions to bag-level via max-pooling.

    At prediction time MIL uses hard max-pooling (this function). During training,
    the LSE objective (:class:`~milgboost.objective.lse.LSE`) uses a soft
    approximation (log-sum-exp) which converges to hard max as ``r -> infinity``.

    Args:
        raw_preds: Instance-level predictions (logits), shape (n_instances,).
        z: Bag IDs, shape (n_instances,).

    Returns:
        Bag-level logits, shape (n_bags,). Sorted by bag_id (ascending).
    """
    unique_z = sorted(np.unique(z))
    bag_logit = []
    for b in unique_z:
        mask = z == b
        instance_preds = raw_preds[mask]
        max_pred = np.max(instance_preds)
        bag_logit.append(max_pred)
    return np.array(bag_logit)


def logistic_sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply logistic sigmoid function element-wise.

    Args:
        x: Input array (logits).

    Returns:
        Probabilities in range (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x))


class BaseMILModel(ABC):
    @abstractmethod
    def fit(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, *args: Any, **kwargs: Any
    ) -> Self: ...

    def fit_bags(self, bags: Sequence[LabeledBag], *args: Any, **kwargs: Any) -> Self:
        x, y, z = labeled_bags_to_arrays(bags)
        return self.fit(x, y, z, *args, **kwargs)

    @abstractmethod
    def predict_proba(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Predict bag-level probabilities.

        Args:
            x: Instance features, shape (n_instances, n_features).
            z: Bag IDs, shape (n_instances,). Each unique ID corresponds to one bag.

        Returns:
            Bag-level probabilities, shape (n_bags,). Output is sorted by bag_id
            (ascending order of unique z values).
        """

    def predict_proba_bags(self, bags: Sequence[Bag]) -> np.ndarray:
        """Predict bag-level probabilities from a sequence of bags.

        Args:
            bags: Sequence of Bag objects.

        Returns:
            Bag-level probabilities, shape (n_bags,). Output is sorted by bag_id
            (ascending order of bag index in the input sequence).
        """
        x, z = bags_to_arrays(bags)
        return self.predict_proba(x, z)

    def predict(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Predict bag-level labels.

        Args:
            x: Instance features, shape (n_instances, n_features).
            z: Bag IDs, shape (n_instances,). Each unique ID corresponds to one bag.

        Returns:
            Bag-level labels (0 or 1), shape (n_bags,). Output is sorted by bag_id
            (ascending order of unique z values).
        """
        proba = self.predict_proba(x, z)
        return (proba >= 0.5).astype(int)

    def predict_bags(self, bags: Sequence[Bag]) -> np.ndarray:
        """Predict bag-level labels from a sequence of bags.

        Args:
            bags: Sequence of Bag objects.

        Returns:
            Bag-level labels (0 or 1), shape (n_bags,). Output is sorted by bag_id
            (ascending order of bag index in the input sequence).
        """
        x, z = bags_to_arrays(bags)
        return self.predict(x, z)
