import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.models.parameter_selection import validate_logreg


class TestThreshold:
    @pytest.mark.parametrize("seed",
                             [np.random.randint(0, 100) for _ in range(500)])
    def test_validate_logreg(self, seed):
        X, y = make_regression(
            n_samples=50, n_features=12, random_state=seed, noise=4.0, bias=100.0
        )
        m = np.mean(y)
        y = (y > m).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        train_score_fr, val_score_fr = validate_logreg(X_train, y_train, X_test, y_test, params={})
        eps = 1e-3
        assert train_score_fr <= 1.0 + eps
        assert val_score_fr <= 1.0 + eps
