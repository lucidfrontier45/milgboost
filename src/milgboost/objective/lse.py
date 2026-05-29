import numpy as np

from milgboost.objective.base import BaseMILObjective


class LSEBCE(BaseMILObjective):
    """LogSumExp Binary Cross Entropy (LSE-BCE) MIL objective.

    A smooth approximation of the max-instance binary cross-entropy loss for
    multiple-instance learning. The bag-level logit is computed via the
    LogSumExp of instance predictions, then passed through a sigmoid for
    binary classification. The parameter ``r`` controls the sharpness of the
    LogSumExp approximation (higher = closer to max).
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
        exp_preds = np.exp(r * preds)
        bag_sum_exp = np.bincount(bag_ids, weights=exp_preds, minlength=num_bags)

        w_ij = exp_preds / (bag_sum_exp[bag_ids] + 1e-12)
        bag_y = (1.0 / r) * np.log(bag_sum_exp + 1e-12)
        bag_p = 1.0 / (1.0 + np.exp(-bag_y))

        p_i = bag_p[bag_ids]
        t_i = y

        grad = (p_i - t_i) * w_ij
        hess = p_i * (1.0 - p_i) * (w_ij**2) + r * (p_i - t_i) * w_ij * (1.0 - w_ij)
        hess = np.clip(hess, 1e-4, None)

        return grad, hess
