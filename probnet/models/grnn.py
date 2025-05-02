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
        The kernel function to use ('gaussian', 'laplace', 'epanechnikov').
    metric : str
        The distance metric to use ('euclidean', 'manhattan', 'cosine').
    k_neighbors : int or None
        The number of nearest neighbors to consider. If None, all training samples are used.
    normalize_output : bool
        Whether to normalize the output predictions.
    kwargs : dict
        Additional keyword arguments for customization.
    """

    def __init__(self, sigma=1.0, kernel='gaussian', metric='euclidean',
                 k_neighbors=None, normalize_output=True, **kwargs):
        """
        Initialize the GRNN regressor.

        Parameters
        ----------
        sigma : float, default=1.0
            The bandwidth parameter for the kernel function.
        kernel : str, default='gaussian'
            The kernel function to use ('gaussian', 'laplace', 'epanechnikov').
        metric : str, default='euclidean'
            The distance metric to use ('euclidean', 'manhattan', 'cosine').
        k_neighbors : int or None, default=None
            The number of nearest neighbors to consider. If None, all training samples are used.
        normalize_output : bool, default=True
            Whether to normalize the output predictions.
        kwargs : dict
            Additional keyword arguments for customization.
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.kernel = kernel
        self.metric = metric
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
            self.nn_ = NearestNeighbors(n_neighbors=self.k_neighbors, metric=self.metric)
            self.nn_.fit(X)

        return self

    def _compute_distance(self, X1, X2):
        """
        Compute the distance between all pairs of X1 and X2.

        Parameters
        ----------
        X1 : array-like of shape (n_samples_1, n_features)
            First set of samples.
        X2 : array-like of shape (n_samples_2, n_features)
            Second set of samples.

        Returns
        -------
        distances : array-like of shape (n_samples_1, n_samples_2)
            Pairwise distances between X1 and X2.
        """
        if self.metric == 'euclidean':
            X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            return np.sqrt(np.maximum(X1_sq - 2 * X1 @ X2.T + X2_sq, 0))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X1[:, None, :] - X2[None, :, :]), axis=2)
        elif self.metric == 'cosine':
            X1_norm = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
            X2_norm = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
            cosine_sim = X1_norm @ X2_norm.T
            return 1 - cosine_sim
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _apply_kernel(self, distances):
        """
        Apply the kernel function to the distance matrix.

        Parameters
        ----------
        distances : array-like of shape (n_samples_1, n_samples_2)
            Pairwise distances.

        Returns
        -------
        weights : array-like of shape (n_samples_1, n_samples_2)
            Kernel-applied weights.
        """
        if self.kernel == 'gaussian':
            return np.exp(- (distances ** 2) / (2 * self.sigma ** 2))
        elif self.kernel == 'laplace':
            return np.exp(- distances / self.sigma)
        elif self.kernel == 'epanechnikov':
            mask = distances <= self.sigma
            w = np.zeros_like(distances)
            w[mask] = 1 - (distances[mask] / self.sigma) ** 2
            return w
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

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
            weights = self._apply_kernel(distances)
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
            distances = self._compute_distance(X, self.X_train_)  # shape (n_test, n_train)
            weights = self._apply_kernel(distances)               # shape (n_test, n_train)

            numerator = weights @ self.y_train_                   # (n_test,)
            if self.normalize_output:
                denominator = np.sum(weights, axis=1)
                return numerator / np.where(denominator == 0, 1e-8, denominator)
            else:
                return numerator

    def score(self, X, y):
        """Return the real R2 (Coefficient of Determination) metric, not (Pearson’s Correlation Index)^2 like Scikit-Learn library.

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
            The result of selected metric
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