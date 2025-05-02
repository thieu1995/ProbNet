#!/usr/bin/env python
# Created by "Thieu" at 14:46, 02/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from probnet import PnnClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from probnet import DataTransformer


def get_cross_val_score(X, y, cv=3):
    """
    Calculate cross-validation scores for a given model and dataset.

    Parameters
    ----------
    model : object
        The model to evaluate.
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Target vector.
    cv : int, optional
        Number of folds in cross-validation (default is 3).

    Returns
    -------
    scores : array, shape (n_splits,)
        Cross-validation scores for each fold.
    """
    ## Train and test
    model = PnnClassifier(sigma=1.0, kernel='laplace', metric='manhattan', normalize_output=True)
    return cross_val_score(model, X, y, cv=cv)


def get_pipe_line(X, y):
    """
    Create a pipeline for the model and dataset.

    Parameters
    ----------
    model : object
        The model to evaluate.
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Target vector.

    Returns
    -------
    pipeline : object
        The pipeline object.
    """
    ## Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    ## Train and test
    model = PnnClassifier(sigma=1.0, kernel='laplace', metric='manhattan', normalize_output=True)

    pipe = Pipeline([
        ("dt", DataTransformer(scaling_methods=("standard", "minmax"))),
        ("pnn", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return model.evaluate(y_true=y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS", "AS", "RS", "PS"])


def get_grid_search(X, y):
    """
    Create a grid search for the model and dataset.

    Parameters
    ----------
    model : object
        The model to evaluate.
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Target vector.

    Returns
    -------
    pipeline : object
        The pipeline object.
    """
    ## Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    para_grid = {
        'sigma': np.linspace(0.1, 5.0, 50),
        'kernel': ('gaussian', 'laplace', "cauchy", "epanechnikov"),
        "metric": ('euclidean', 'manhattan'),
    }

    ## Create a gridsearch
    model = PnnClassifier(normalize_output=True)
    clf = GridSearchCV(model, para_grid, cv=5, scoring='accuracy', verbose=2)
    clf.fit(X_train, y_train)
    print("Best parameters found: ", clf.best_params_)
    print("Best model: ", clf.best_estimator_)
    print("Best training score: ", clf.best_score_)

    ## Predict
    y_pred = clf.predict(X_test)
    return model.evaluate(y_true=y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS", "AS", "RS", "PS"])


## Load data object
X, y = load_breast_cancer(return_X_y=True)

print(get_cross_val_score(X, y, cv=3))
print(get_pipe_line(X, y))
print(get_grid_search(X, y))
