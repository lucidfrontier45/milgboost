import numpy as np
import pytest

from milgboost.objective.lse import LSEBCE


def test_lse_basic() -> None:
    y = np.array([1, 0, 1, 0])
    bag_ids = np.array([0, 0, 1, 1])
    preds = np.array([0.5, -0.5, 1.0, -1.0])
    r = 1.0

    objective = LSEBCE(r=r)
    grad, hess = objective(y, bag_ids, preds)

    assert grad.shape == preds.shape
    assert hess.shape == preds.shape
    assert hess.min() >= 1e-4


def test_lse_single_bag() -> None:
    y = np.array([1, 0, 1])
    bag_ids = np.array([0, 0, 0])
    preds = np.array([1.0, 0.0, -1.0])
    r = 1.0

    objective = LSEBCE(r=r)
    grad, hess = objective(y, bag_ids, preds)

    exp_preds = np.exp(preds)
    bag_sum_exp = np.sum(exp_preds)
    w_ij = exp_preds / bag_sum_exp
    bag_p = 1.0 / (1.0 + np.exp(-np.log(bag_sum_exp)))
    expected_grad = (bag_p - y) * w_ij

    assert grad.shape == preds.shape
    assert hess.shape == preds.shape
    assert np.allclose(grad, expected_grad)


def test_lse_binary_labels() -> None:
    y = np.array([0, 1, 0, 1])
    bag_ids = np.array([0, 0, 1, 1])
    preds = np.array([0.0, 0.0, 0.0, 0.0])
    r = 1.0

    objective = LSEBCE(r=r)
    grad, hess = objective(y, bag_ids, preds)

    assert grad.shape == preds.shape
    assert hess.shape == preds.shape
    assert np.all(hess > 0)


def test_lse_r_parameter_effect() -> None:
    y = np.array([1, 0])
    bag_ids = np.array([0, 0])
    preds = np.array([1.0, -1.0])

    _, hess_r1 = LSEBCE(r=1.0)(y, bag_ids, preds)
    _, hess_r2 = LSEBCE(r=2.0)(y, bag_ids, preds)

    assert not np.allclose(hess_r1, hess_r2)


def test_lse_hessian_positive() -> None:
    y = np.array([1, 0, 1, 0, 1])
    bag_ids = np.array([0, 0, 1, 1, 2])
    preds = np.array([2.0, -2.0, 3.0, -3.0, 0.5])
    r = 1.0

    objective = LSEBCE(r=r)
    _, hess = objective(y, bag_ids, preds)

    assert np.all(hess > 0)


def test_lse_known_values() -> None:
    y = np.array([1.0, 0.0])
    bag_ids = np.array([0, 0])
    preds = np.array([0.0, 0.0])
    r = 1.0

    objective = LSEBCE(r=r)
    grad, hess = objective(y, bag_ids, preds)

    bag_sum_exp = 2.0
    bag_y = np.log(bag_sum_exp)
    bag_p = 1.0 / (1.0 + np.exp(-bag_y))
    w_ij = np.array([0.5, 0.5])

    expected_grad = (bag_p - y) * w_ij

    assert np.allclose(grad, expected_grad)
    assert hess.min() >= 1e-4


def test_lse_non_dense_bag_ids_raises() -> None:
    y = np.array([1.0, 0.0, 1.0])
    bag_ids = np.array([0, 2, 2])  # gap: bag id 1 missing
    preds = np.array([0.5, -0.5, 1.0])
    objective = LSEBCE(r=1.0)
    with pytest.raises(ValueError, match="dense 0-based contiguous"):
        objective(y, bag_ids, preds)
