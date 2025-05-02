#!/usr/bin/env python
# Created by "Thieu" at 11:13, 02/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from probnet.models.base_net import BaseNet


class PnnClassifier(BaseNet, ClassifierMixin):
    def __init__(self, sigma=1.0, kernel='gaussian', metric='euclidean',
                 normalize_output=True, class_prior=None, sample_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.kernel = kernel
        self.metric = metric
        self.normalize_output = normalize_output
        self.class_prior = class_prior
        self.sample_weights = sample_weights

    def _compute_distance(self, X1, X2):
        """Vectorized distance computation"""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=-1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :]), axis=-1)
        else:
            raise ValueError("Unsupported metric: choose 'euclidean' or 'manhattan'")

    def _apply_kernel(self, dists):
        """Apply chosen kernel function"""
        if self.kernel == 'gaussian':
            return np.exp(- (dists ** 2) / (2 * self.sigma ** 2))
        elif self.kernel == 'laplace':
            return np.exp(- dists / self.sigma)
        elif self.kernel == 'cauchy':
            return 1 / (1 + (dists ** 2) / (self.sigma ** 2))
        elif self.kernel == 'epanechnikov':
            u = dists / self.sigma
            k = 0.75 * (1 - u ** 2)
            k[u > 1] = 0
            return k
        else:
            raise ValueError("Unsupported kernel")

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.n_labels = len(self.classes_)
        self.class_indices_ = {cls: np.where(y == cls)[0] for cls in self.classes_}
        return self

    def _estimate_class_scores(self, X):
        distances = self._compute_distance(X, self.X_train_)
        kernel_values = self._apply_kernel(distances)

        probs = np.zeros((X.shape[0], len(self.classes_)))
        for idx, cls in enumerate(self.classes_):
            indices = self.class_indices_[cls]
            weights = np.ones(len(indices)) if self.sample_weights is None else self.sample_weights[indices]
            weighted_kernels = kernel_values[:, indices] * weights
            probs[:, idx] = np.sum(weighted_kernels, axis=1)
            if self.class_prior is not None:
                probs[:, idx] *= self.class_prior[idx]

        if self.normalize_output:
            row_sums = np.sum(probs, axis=1, keepdims=True)
            probs = np.divide(probs, row_sums, where=row_sums != 0)

        return probs

    def predict(self, X):
        X = check_array(X)
        probs = self._estimate_class_scores(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        X = check_array(X)
        return self._estimate_class_scores(X)

    def score(self, X, y):
        """Return the real Accuracy Score metric

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
        return self._BaseNet__score_cls(X, y, "AS")

    def scores(self, X, y, list_metrics=("AS", "RS")):
        """Return the list of classification metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
           ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
           True values for `X`.

        list_metrics : list, default=("AS", "RS")
           You can get classification metrics from Permetrics library: https://permetrics.readthedocs.io/en/latest/pages/classification.html

        Returns
        -------
        results : dict
           The results of the list metrics
        """
        return self._BaseNet__scores_cls(X, y, list_metrics)

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """Return the list of classification performance metrics of the prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list
            You can get classification metrics from Permetrics library: https://permetrics.readthedocs.io/en/latest/pages/classification.html

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseNet__evaluate_cls(y_true, y_pred, list_metrics)
