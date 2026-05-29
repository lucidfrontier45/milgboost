import abc

import numpy as np


class BaseMILObjective(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        y: np.ndarray,
        bag_ids: np.ndarray,
        preds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...
