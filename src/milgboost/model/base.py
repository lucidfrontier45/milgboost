from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Self

import numpy as np

from ..types import Bag, LabeledBag, bags_to_arrays, labeled_bags_to_arrays


class BaseMILModel(ABC):
    @abstractmethod
    def fit(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, *args: Any, **kwargs: Any
    ) -> Self: ...

    def fit_bags(self, bags: Sequence[LabeledBag], *args, **kwargs):
        x, y, z = labeled_bags_to_arrays(bags)
        self.fit(x, y, z, *args, **kwargs)

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
