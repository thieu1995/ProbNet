#!/usr/bin/env python
# Created by "Thieu" at 08:04, 03/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from probnet.models.grnn import GrnnRegressor


@pytest.fixture
def synthetic_dataset():
    X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_fit_and_predict(synthetic_dataset):
    X_train, X_test, y_train, y_test = synthetic_dataset

    reg = GrnnRegressor(sigma=1.0, kernel='gaussian', dist='euclidean', normalize_output=True)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_test.shape


def test_score_function(synthetic_dataset):
    X_train, X_test, y_train, y_test = synthetic_dataset

    reg = GrnnRegressor(sigma=1.0, kernel='gaussian', dist='euclidean', normalize_output=True)
    reg.fit(X_train, y_train)
    score = reg.score(X_test, y_test)

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_invalid_kernel():
    with pytest.raises(ValueError):
        reg = GrnnRegressor(kernel='invalid_kernel')
        reg.fit(np.random.rand(10, 4), np.random.rand(10))


def test_invalid_metric():
    with pytest.raises(ValueError):
        reg = GrnnRegressor(dist='invalid_metric')
        reg.fit(np.random.rand(10, 4), np.random.rand(10))


def test_predict_without_fit():
    reg = GrnnRegressor(sigma=1.0, kernel='gaussian', dist='euclidean', normalize_output=True)
    with pytest.raises(ValueError):
        reg.predict(np.random.rand(5, 4))
