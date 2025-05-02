#!/usr/bin/env python
# Created by "Thieu" at 11:03, 02/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator
from permetrics import RegressionMetric, ClassificationMetric
from probnet.helpers import validator
from probnet.helpers.metrics import get_all_regression_metrics, get_all_classification_metrics


class BaseNet(BaseEstimator):
    """
    class BaseNet(BaseEstimator):
        A base class for implementing Extreme Learning Machines (ELM) with support for both classification and regression tasks.

        Attributes
        ----------
        layer_sizes : list
            List containing the sizes of each layer in the network.

        act_name : str
            The name of the activation function to be used.

        network : object
            The ELM network object.

        loss_train : list
            List of loss values recorded during training.

        n_labels : int
            Number of labels in the dataset.
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()
    CLS_OBJ_LOSSES = {}

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.n_labels = None

    @staticmethod
    def _check_method(method=None, list_supported_methods=None):
        if type(method) is str:
            return validator.check_str("method", method, list_supported_methods)
        else:
            raise ValueError(f"method should be a string and belongs to {list_supported_methods}")

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
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
        metric = self._check_method(metric, list(self.SUPPORTED_CLS_METRICS.keys()))
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

