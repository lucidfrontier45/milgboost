from typing import Any, Self

import lightgbm as lgb
import numpy as np

from milgboost.objective import BaseMILObjective

from ..types import validate_dense_bag_ids
from .base import BaseMILModel, instance_max_pooling, logistic_sigmoid


class _LightGBMMILObjective:
    def __init__(self, base_objective: BaseMILObjective, bag_ids: np.ndarray) -> None:
        self._base_obj = base_objective
        self._bag_ids = bag_ids

    def __call__(
        self, preds: np.ndarray, train_data: lgb.Dataset
    ) -> tuple[np.ndarray, np.ndarray]:
        y = train_data.get_label()
        bag_ids = np.asarray(self._bag_ids, dtype=np.int64)
        return self._base_obj(np.asarray(y, dtype=np.float64), bag_ids, preds)


class LightGBMMILModel(BaseMILModel):
    def __init__(
        self,
        objective: BaseMILObjective,
        lgb_params: dict | None = None,
        num_boost_round: int = 100,
    ) -> None:
        self._base_objective = objective
        self._lgb_params = lgb_params
        self._num_boost_round = num_boost_round

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        n_bags = validate_dense_bag_ids(z)
        if len(y) != n_bags:
            raise ValueError(
                f"y length {len(y)} != number of bags {n_bags} implied by z"
            )

        instance_labels = y[z.astype(np.int64)]
        dtrain = lgb.Dataset(x, label=instance_labels)

        params: dict[str, Any] = (
            dict(self._lgb_params) if self._lgb_params else {"boosting_type": "gbdt"}
        )
        params.update(kwargs)
        params["objective"] = _LightGBMMILObjective(
            self._base_objective, np.asarray(z, dtype=np.int64)
        )

        self.model_ = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=self._num_boost_round,
        )

        return self

    def predict_proba(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Predict bag-level probabilities.

        Aggregation uses hard max-pooling (see
        :func:`~milgboost.model.base.instance_max_pooling`), which differs from
        the soft approximation (LSE) used during training.

        Args:
            x: Instance features, shape (n_instances, n_features).
            z: Bag IDs, shape (n_instances,). Each unique ID corresponds to one bag.

        Returns:
            Bag-level probabilities, shape (n_bags,). Output is sorted by bag_id
            (ascending order of unique z values).
        """
        raw_preds = np.asarray(self.model_.predict(x), dtype=np.float64)
        bag_logits = instance_max_pooling(raw_preds, z)
        return logistic_sigmoid(bag_logits)
