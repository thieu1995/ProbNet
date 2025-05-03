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
    """
    Probabilistic Neural Network (PNN) Classifier implementation.

    Attributes
    ----------
    sigma : float
        The bandwidth parameter for the kernel function.
    kernel : str
        The kernel function to use ('gaussian', 'laplace', 'cauchy', 'epanechnikov',...).
    dist : str
        The distance method to use ('euclidean', 'manhattan',...).
    normalize_output : bool
        Whether to normalize the output probabilities.
    class_prior : array-like or None
        Prior probabilities of the classes. If None, uniform priors are used.
    sample_weights : array-like or None
        Weights for the training samples. If None, all samples are weighted equally.
    kwargs : dict
        Additional keyword arguments for customization.
    """

    SUPPORTED_KERNELS = ['gaussian', 'laplace', 'cauchy', 'epanechnikov']
    SUPPORTED_METRICS = ['euclidean', 'manhattan']

    def __init__(self, sigma=1.0, kernel='gaussian', dist='euclidean',
                 normalize_output=True, class_prior=None, sample_weights=None, **kwargs):
        """
        Initialize the PNN Classifier.

        Parameters
        ----------
        sigma : float, default=1.0
            The bandwidth parameter for the kernel function.
        kernel : str, default='gaussian'
            The kernel function to use ('gaussian', 'laplace', 'cauchy', 'epanechnikov',...).
        dist : str, default='euclidean'
            The distance method to use ('euclidean', 'manhattan',...).
        normalize_output : bool, default=True
            Whether to normalize the output probabilities.
        class_prior : array-like or None, default=None
            Prior probabilities of the classes. If None, uniform priors are used.
        sample_weights : array-like or None, default=None
            Weights for the training samples. If None, all samples are weighted equally.
        kwargs : dict
            Additional keyword arguments for customization.
        """
        super().__init__(sigma, kernel, dist, **kwargs)
        self.normalize_output = normalize_output
        self.class_prior = class_prior
        self.sample_weights = sample_weights

    def fit(self, X, y):
        """
        Fit the PNN model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : PnnClassifier
            The fitted model.
        """
        X, y = check_X_y(X, y)
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.n_labels = len(self.classes_)
        self.class_indices_ = {cls: np.where(y == cls)[0] for cls in self.classes_}
        return self

    def _estimate_class_scores(self, X):
        """
        Estimate class scores for the given input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        probs : array-like of shape (n_samples, n_classes)
            Estimated class scores.
        """
        distances = self.dist_func(X, self.X_train_)
        kernel_values = self.kernel_func(distances, self.sigma)

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
        """
        Predict class labels for the given input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        predictions : array-like of shape (n_samples,)
            Predicted class labels.
        """
        X = check_array(X)
        probs = self._estimate_class_scores(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities for the given input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        probabilities : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
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
