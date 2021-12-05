import pytest
import numpy as np

from src.models.threshold_tuning import _get_optimal_threshold_by_accuracy


class TestThreshold:
    @pytest.mark.parametrize("threshold_true",
                             [np.random.random() for _ in range(500)])
    def test_get_optimal_threshold_by_accuracy(self, threshold_true):
        probs = np.random.random(100)
        prob_0 = probs[probs < threshold_true]
        prob_1 = probs[probs >= threshold_true]

        prob_0 = np.hstack([prob_0, [0.0]])
        prob_1 = np.hstack([prob_1, [1.0]])

        best_threshold, best_accuracy, prob_list, acc_list = _get_optimal_threshold_by_accuracy(
            np.hstack([prob_0, prob_1]),
            [0] * len(prob_0) + [1] * len(prob_1)
        )
        eps = 1e-3
        assert np.max(prob_0) - eps <= best_threshold <= np.min(prob_1) + eps
