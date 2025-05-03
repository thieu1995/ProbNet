#!/usr/bin/env python
# Created by "Thieu" at 19:28, 02/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from probnet.models.base_net import BaseNet


class GrnnRegressor(BaseNet, RegressorMixin):
    """
    General Regression Neural Network (GRNN) implementation.

    Attributes
    ----------
    sigma : float
        The bandwidth parameter for the kernel function.
    kernel : str
        The kernel function to use ('gaussian', 'laplace', 'epanechnikov',...).
    dist : str
        The distance method to use ('euclidean', 'manhattan', 'cosine',...).
    k_neighbors : int or None
        The number of nearest neighbors to consider. If None, all training samples are used.
    normalize_output : bool
        Whether to normalize the output predictions.
    kwargs : dict
        Additional keyword arguments for customization.
    """

    SUPPORTED_KERNELS = ['gaussian', 'laplace', 'cauchy', 'epanechnikov']
    SUPPORTED_METRICS = ['euclidean', 'manhattan']

    def __init__(self, sigma=1.0, kernel='gaussian', dist='euclidean',
                 k_neighbors=None, normalize_output=True, **kwargs):
        """
        Initialize the GRNN regressor.

        Parameters
        ----------
        sigma : float, default=1.0
            The bandwidth parameter for the kernel function.
        kernel : str, default='gaussian'
            The kernel function to use ('gaussian', 'laplace', 'epanechnikov',...).
        dist : str, default='euclidean'
            The distance method to use ('euclidean', 'manhattan', 'cosine',...).
        k_neighbors : int or None, default=None
            The number of nearest neighbors to consider. If None, all training samples are used.
        normalize_output : bool, default=True
            Whether to normalize the output predictions.
        kwargs : dict
            Additional keyword arguments for customization.
        """
        super().__init__(sigma, kernel, dist, **kwargs)
        self.k_neighbors = k_neighbors
        self.normalize_output = normalize_output

    def fit(self, X, y):
        """
        Fit the GRNN model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GrnnRegressor
            The fitted model.
        """
        X, y = check_X_y(X, y)
        self.X_train_ = X
        self.y_train_ = y

        # Dùng NearestNeighbors nếu có k_neighbors
        if self.k_neighbors is not None:
            self.nn_ = NearestNeighbors(n_neighbors=self.k_neighbors, metric=self.dist)
            self.nn_.fit(X)

        return self

    def predict(self, X):
        """
        Predict target values for the given input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        predictions : array-like of shape (n_samples,)
            Predicted target values.
        """
        check_is_fitted(self, ["X_train_", "y_train_"])
        X = check_array(X)

        if self.k_neighbors is not None:
            # Chỉ dùng k hàng xóm gần nhất
            distances, indices = self.nn_.kneighbors(X, return_distance=True)
            weights = self.kernel_func(distances, self.sigma)
            preds = []
            for i in range(X.shape[0]):
                y_neighbors = self.y_train_[indices[i]]
                w = weights[i]
                if self.normalize_output:
                    w_sum = np.sum(w)
                    if w_sum == 0:
                        preds.append(0.0)
                    else:
                        preds.append(np.dot(w, y_neighbors) / w_sum)
                else:
                    preds.append(np.dot(w, y_neighbors))
            return np.array(preds)

        else:
            # Dùng toàn bộ dữ liệu huấn luyện
            distances = self.dist_func(X, self.X_train_)  # shape (n_test, n_train)
            weights = self.kernel_func(distances, self.sigma)               # shape (n_test, n_train)

            numerator = weights @ self.y_train_                   # (n_test,)
            if self.normalize_output:
                denominator = np.sum(weights, axis=1)
                return numerator / np.where(denominator == 0, 1e-8, denominator)
            else:
                return numerator

    def score(self, X, y):
        """Return the real R2 (Coefficient of Determination) method, not (Pearson’s Correlation Index)^2 like Scikit-Learn library.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        Returns
        -------
        result : float
            The result of selected method
        """
        return self._BaseNet__score_reg(X, y, "R2")

    def scores(self, X, y, list_metrics=("MSE", "MAE")):
        """Return the list of regression metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_metrics : list, default=("MSE", "MAE")
            You can get regression metrics from Permetrics library: https://permetrics.readthedocs.io/en/latest/pages/regression.html

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseNet__scores_reg(X, y, list_metrics)

    def evaluate(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        """Return the list of performance metrics of the prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseNet__evaluate_reg(y_true, y_pred, list_metrics)
