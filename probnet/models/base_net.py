#!/usr/bin/env python
# Created by "Thieu" at 11:03, 02/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import inspect
import pickle
import pprint
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator
from permetrics import RegressionMetric, ClassificationMetric
from probnet.helpers import validator
from probnet.helpers import kernel as kernel_module
from probnet.helpers import distance as distance_module
from probnet.helpers.metrics import get_all_regression_metrics, get_all_classification_metrics


class BaseNet(BaseEstimator):
    """
    Base class for neural network models. Inherits from `BaseEstimator` to integrate with scikit-learn pipelines.

    Attributes
    ----------
    SUPPORTED_CLS_METRICS : dict
        Dictionary of supported classification metrics.
    SUPPORTED_REG_METRICS : dict
        Dictionary of supported regression metrics.
    CLS_OBJ_LOSSES : dict
        Dictionary of classification objective losses.
    SUPPORTED_KERNELS : list
        List of supported kernel functions.
    SUPPORTED_METRICS : list
        List of supported distance metrics.

    Parameters
    ----------
    sigma : float, default=1.0
        The bandwidth parameter for the kernel function.
    kernel : str, default='gaussian'
        The kernel function to use.
    dist : str, default='euclidean'
        The distance metric to use.
    kwargs : dict
        Additional keyword arguments for customization.
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()
    CLS_OBJ_LOSSES = {}

    SUPPORTED_KERNELS = ["gaussian", "laplace", "cauchy", "epanechnikov", "uniform",
                         "triangular", "quartic", "cosine", "logistic", "sigmoid",
                         "multiquadric", "inverse_multiquadric", "rational_quadratic",
                         "exponential", "power", "linear", "bessel", "vonmises", "vonmises_fisher"]
    SUPPORTED_METRICS = ['euclidean', 'manhattan', "chebyshev", "minkowski", "hamming", "canberra",
                         "braycurtis", "jaccard", "sokalmichener", "sokalsneath", "russellrao",
                         "yule", "kulsinski", "rogers_tanimoto", "kulczynski", "morisita", "morisita_horn",
                         "dice", "kappa", "rogers", "jensen", "jensen_shannon", "hellinger",
                         "bhattacharyya", "cityblock", "cosin", "correlation", "mahalanobis"]

    def __init__(self, sigma=1.0, kernel='gaussian', dist='euclidean', **kwargs):
        """
        Initialize the BaseNet class.

        Parameters
        ----------
        sigma : float, default=1.0
            The bandwidth parameter for the kernel function.
        kernel : str, default='gaussian'
            The kernel function to use.
        dist : str, default='euclidean'
            The distance metric to use.
        kwargs : dict
            Additional keyword arguments for customization.
        """
        super().__init__()
        self.sigma = sigma
        self.set_kernel(kernel)
        self.set_dist(dist)
        self.kwargs = kwargs
        self.n_labels = None

    def set_kernel(self, kernel):
        """
        Set the kernel function.

        Parameters
        ----------
        kernel : str
            The kernel function to use ('gaussian', 'laplace', 'cauchy', 'epanechnikov',...).
        """
        self.kernel = validator.check_str("kernel", kernel, self.SUPPORTED_KERNELS)
        self.kernel_func = getattr(kernel_module, f"{self.kernel}_kernel")

    def set_dist(self, dist):
        """
        Set the distance metric.

        Parameters
        ----------
        dist : str
            The distance metric to use ('euclidean', 'manhattan',...).
        """
        self.dist = validator.check_str("dist", dist, self.SUPPORTED_METRICS)
        self.dist_func = getattr(distance_module, f"{self.dist}_distance")

    def __repr__(self, **kwargs):
        """Pretty-print parameters like scikit-learn's Estimator.
        """
        param_order = list(inspect.signature(self.__init__).parameters.keys())
        param_dict = {k: getattr(self, k) for k in param_order}

        param_str = ", ".join(f"{k}={repr(v)}" for k, v in param_dict.items())
        if len(param_str) <= 80:
            return f"{self.__class__.__name__}({param_str})"
        else:
            formatted_params = ",\n  ".join(f"{k}={pprint.pformat(v)}" for k, v in param_dict.items())
            return f"{self.__class__.__name__}(\n  {formatted_params}\n)"

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        pass

    def predict(self, X):
        """
        Predict target values for the given input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        array-like
            Predicted target values.
        """
        pass

    def predict_proba(self, X):
        """
        Predict class probabilities for the given input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        array-like
            Predicted class probabilities.
        """
        pass

    def __evaluate_reg(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        """
        Parameters
        ----------
        y_true : array-like
            Ground truth (correct) target values.
        y_pred : array-like
            Estimated target values.
        list_metrics : tuple/list of str, optional
            List of metric names to evaluate. Default is ("MSE", "MAE").
        """
        rm = RegressionMetric(y_true=y_true, y_pred=y_pred)
        return rm.get_metrics_by_list_names(list_metrics)

    def __evaluate_cls(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        y_pred : array-like of shape (n_samples,)
            Predicted class labels by the classifier.
        list_metrics : tuple/list of str, optional
            List of metric names to evaluate, by default ("AS", "RS").
        """
        cm = ClassificationMetric(y_true, y_pred)
        return cm.get_metrics_by_list_names(list_metrics)

    def __score_reg(self, X, y, metric="RMSE"):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples used for prediction.

        y : array-like of shape (n_samples,)
            The true target values.

        method : str, optional, default="RMSE"
            The regression metric to be used for scoring. Must be one of the supported metrics in SUPPORTED_REG_METRICS.

        Returns
        -------
        float
            The calculated regression metric based on the method provided.
        """
        y_pred = self.predict(X)
        return RegressionMetric(y, y_pred).get_metric_by_name(metric)[metric]

    def __scores_reg(self, X, y, list_metrics=("MSE", "MAE")):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            True values for X.

        list_metrics : tuple of str, optional
            List of evaluation metrics to be used. Default is ("MSE", "MAE").
        """
        y_pred = self.predict(X)
        return self.__evaluate_reg(y_true=y, y_pred=y_pred, list_metrics=list_metrics)

    def __score_cls(self, X, y, metric="AS"):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples to score.

        y : array-like of shape (n_samples,)
            True labels for X.

        method : str, default="AS"
            Scoring method to use. Supported methods are determined by the keys in self.SUPPORTED_CLS_METRICS.

        Returns
        -------
        float
            Computed score based on the specified method.
        """
        metric = validator.check_str("metric", metric, list(self.SUPPORTED_CLS_METRICS.keys()))
        return_prob = False
        if self.n_labels > 2:
            if metric in self.CLS_OBJ_LOSSES:
                return_prob = True
        if return_prob:
            y_pred = self.predict_proba(X)
        else:
            y_pred = self.predict(X)
        cm = ClassificationMetric(y_true=y, y_pred=y_pred)
        return cm.get_metric_by_name(metric)[metric]

    def __scores_cls(self, X, y, list_metrics=("AS", "RS")):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for the samples for which predictions are to be made.

        y : array-like of shape (n_samples,)
            True labels for the samples.

        list_metrics : tuple of str, optional
            List of method names to evaluate. Possible values include 'AS', 'RS', etc. Default is ('AS', 'RS').

        Returns
        -------
        dict
            A dictionary with the performance metrics from the selected methods listed in `list_metrics`.
        """
        list_errors = list(set(list_metrics) & set(self.CLS_OBJ_LOSSES))
        list_scores = list((set(self.SUPPORTED_CLS_METRICS.keys()) - set(self.CLS_OBJ_LOSSES)) & set(list_metrics))
        t1 = {}
        if len(list_errors) > 0:
            return_prob = False
            if self.n_labels > 2:
                return_prob = True
            if return_prob:
                y_pred = self.predict_proba(X)
            else:
                y_pred = self.predict(X)
            t1 = self.__evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_errors)
        y_pred = self.predict(X)
        t2 = self.__evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_scores)
        return {**t2, **t1}

    def score(self, X, y):
        """Default interface for score function"""
        pass

    def scores(self, X, y, list_metrics=None):
        """Return the list of metrics of the prediction."""
        pass

    def evaluate(self, y_true, y_pred, list_metrics=None):
        """Return the list of performance metrics of the prediction."""
        pass

    def save_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv"):
        """
        Save evaluation metrics to csv file

        Parameters
        ----------
        y_true : ground truth data
        y_pred : predicted output
        list_metrics : list of evaluation metrics
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results = self.evaluate(y_true, y_pred, list_metrics)
        df = pd.DataFrame.from_dict(results, orient='index').T
        df.to_csv(f"{save_path}/{filename}", index=False)

    def save_y_predicted(self, X, y_true, save_path="history", filename="y_predicted.csv"):
        """
        Save the predicted results to csv file

        Parameters
        ----------
        X : The features data, nd.ndarray
        y_true : The ground truth data
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        y_pred = self.predict(X)
        data = {"y_true": np.squeeze(np.asarray(y_true)), "y_pred": np.squeeze(np.asarray(y_pred))}
        pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_model(self, save_path="history", filename="model.pkl"):
        """
        Save model to pickle file

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".pkl" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        pickle.dump(self, open(f"{save_path}/{filename}", 'wb'))

    @staticmethod
    def load_model(load_path="history", filename="model.pkl"):
        """
        Parameters
        ----------
        load_path : str, optional
            Directory path where the model file is located. Defaults to "history".
        filename : str
            Name of the file to be loaded. If the filename doesn't end with ".pkl", the extension is automatically added.

        Returns
        -------
        object
            The model loaded from the specified pickle file.
        """
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))
