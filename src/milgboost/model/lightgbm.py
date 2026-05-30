from typing import Any, Self

import lightgbm as lgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from milgboost.objective import BaseMILObjective

from .base import BaseMILModel


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


class LightGBMMILModel(BaseMILModel, BaseEstimator, ClassifierMixin):
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
        instance_labels = y[z.astype(np.int64)]
        dtrain = lgb.Dataset(x, label=instance_labels)

        params = self._lgb_params or {"boosting_type": "gbdt", **kwargs}
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
        raw_preds = np.asarray(self.model_.predict(x), dtype=np.float64)
        unique_z = sorted(np.unique(z))
        bag_logit = []

        for b in unique_z:
            mask = z == b
            instance_preds = raw_preds[mask]
            max_pred = np.max(instance_preds)
            bag_logit.append(max_pred)

        return 1.0 / (1.0 + np.exp(-np.array(bag_logit)))
