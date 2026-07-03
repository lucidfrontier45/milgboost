import numpy as np

from milgboost.objective.base import BaseMILObjective


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Branch-wise stable sigmoid avoiding overflow in exp(-x) for x << 0."""
    pos = x >= 0
    result = np.empty_like(x)
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    result[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return result


class LSEBCE(BaseMILObjective):
    """LogSumExp Binary Cross Entropy (LSE-BCE) MIL objective.

    A smooth approximation of the max-instance binary cross-entropy loss for
    multiple-instance learning. The bag-level logit is computed via the
    LogSumExp of instance predictions, then passed through a sigmoid for
    binary classification. The parameter ``r`` controls the sharpness of the
    LogSumExp approximation (higher = closer to max).

    The forward pass uses the max-shift trick (logsumexp = max + log(sum(exp(x - max))))
    to avoid overflow in ``exp(r * preds)`` for large ``preds`` or ``r``.
    """

    def __init__(self, r: float = 1.0) -> None:
        self.r = r

    def __call__(
        self,
        y: np.ndarray,
        bag_ids: np.ndarray,
        preds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        r = self.r
        num_bags = int(np.max(bag_ids)) + 1

        r_preds = r * preds
        bag_max = np.zeros(num_bags, dtype=preds.dtype)
        np.maximum.at(bag_max, bag_ids, r_preds)

        shifted = np.exp(r_preds - bag_max[bag_ids])
        bag_sum_exp = np.bincount(bag_ids, weights=shifted, minlength=num_bags)

        bag_y = (bag_max + np.log(bag_sum_exp)) / r
        bag_p = _stable_sigmoid(bag_y)

        p_i = bag_p[bag_ids]
        t_i = y

        w_ij = shifted / bag_sum_exp[bag_ids]

        grad = (p_i - t_i) * w_ij
        hess = p_i * (1.0 - p_i) * (w_ij**2) + r * (p_i - t_i) * w_ij * (1.0 - w_ij)
        hess = np.clip(hess, 1e-4, None)

        return grad, hess
