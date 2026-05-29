import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from milgboost.datasets import make_mil_data
from milgboost.model import LightGBMMILModel, XGBoostMILModel
from milgboost.objective import LSEBCE
from milgboost.types import LabeledBag, labeled_bags_to_arrays


def split_bags(
    bags: list[LabeledBag], test_ratio: float, seed: int
) -> tuple[list[LabeledBag], list[LabeledBag]]:
    n = len(bags)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    split_point = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split_point], indices[split_point:]
    return [bags[i] for i in train_idx], [bags[i] for i in test_idx]


@pytest.fixture(scope="module")
def mil_data():
    seed = 42
    bags = make_mil_data(
        n_bags=1000,
        n_features=50,
        n_instances=(3, 3),
        random_state=seed,
        informative_ratio=0.1,
        redundant_ratio=0.3,
        n_clusters_per_class=5,
        class_sep=0.1,
        noise=0.2,
    )
    train_bags, test_bags = split_bags(bags, test_ratio=0.2, seed=seed)
    return train_bags, test_bags, seed


def test_lightgbm_mil(mil_data):
    train_bags, test_bags, seed = mil_data

    x_train, y_train, z_train = labeled_bags_to_arrays(train_bags)
    x_test, y_test, z_test = labeled_bags_to_arrays(test_bags)

    model = LightGBMMILModel(objective=LSEBCE(r=1.0), num_boost_round=100)
    model.fit(x_train, y_train, z_train)
    y_proba = model.predict_proba(x_test, z_test)

    assert len(y_proba) == len(y_test)
    assert np.all((y_proba >= 0) & (y_proba <= 1))

    ap = average_precision_score(y_test, y_proba)
    assert ap >= 0.6


def test_xgboost_mil(mil_data):
    train_bags, test_bags, _ = mil_data

    x_train, y_train, z_train = labeled_bags_to_arrays(train_bags)
    x_test, y_test, z_test = labeled_bags_to_arrays(test_bags)

    model = XGBoostMILModel(objective=LSEBCE(r=1.0), num_boost_round=100)
    model.fit(x_train, y_train, z_train)
    y_proba = model.predict_proba(x_test, z_test)

    assert len(y_proba) == len(y_test)
    assert np.all((y_proba >= 0) & (y_proba <= 1))

    ap = average_precision_score(y_test, y_proba)
    assert ap >= 0.6
