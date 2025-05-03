#!/usr/bin/env python
# Created by "Thieu" at 23:14, 02/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


def euclidean_distance(x1, x2):
    """
    Compute the Euclidean distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=-1))


def manhattan_distance(x1, x2):
    """
    Compute the Manhattan distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Manhattan distance between x1 and x2.
    """
    return np.sum(np.abs(x1[:, np.newaxis, :] - x2[np.newaxis, :, :]), axis=-1)


def chebyshev_distance(x1, x2):
    """
    Compute the Chebyshev distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Chebyshev distance between x1 and x2.
    """
    return np.max(np.abs(x1[:, np.newaxis, :] - x2[np.newaxis, :, :]), axis=-1)


def minkowski_distance(x1, x2, p=3):
    """
    Compute the Minkowski distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.
    p : int, optional
        The order of the norm. Default is 3.

    Returns
    -------
    distance : float
        The Minkowski distance between x1 and x2.
    """
    return np.sum(np.abs(x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** p, axis=-1) ** (1/p)


def hamming_distance(x1, x2):
    """
    Compute the Hamming distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Hamming distance between x1 and x2.
    """
    return np.sum(x1[:, np.newaxis, :] != x2[np.newaxis, :, :], axis=-1)


def canberra_distance(x1, x2):
    """
    Compute the Canberra distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Canberra distance between x1 and x2.
    """
    return np.sum(np.abs(x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) / (np.abs(x1[:, np.newaxis, :]) + np.abs(x2[np.newaxis, :, :])), axis=-1)


def braycurtis_distance(x1, x2):
    """
    Compute the Bray-Curtis distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Bray-Curtis distance between x1 and x2.
    """
    return np.sum(np.abs(x1[:, np.newaxis, :] - x2[np.newaxis, :, :]), axis=-1) / np.sum(np.abs(x1[:, np.newaxis, :] + x2[np.newaxis, :, :]), axis=-1)


def jaccard_distance(x1, x2):
    """
    Compute the Jaccard distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Jaccard distance between x1 and x2.
    """
    return 1 - np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1)


def sokalmichener_distance(x1, x2):
    """
    Compute the Sokal-Michener distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Sokal-Michener distance between x1 and x2.
    """
    return 1 - (np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1))


def sokalsneath_distance(x1, x2):
    """
    Compute the Sokal-Sneath distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Sokal-Sneath distance between x1 and x2.
    """
    return 1 - (np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]) + np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1))


def russellrao_distance(x1, x2):
    """
    Compute the Russell-Rao distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Russell-Rao distance between x1 and x2.
    """
    return 1 - (np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1))


def yule_distance(x1, x2):
    """
    Compute the Yule distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Yule distance between x1 and x2.
    """
    return 1 - (np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1))


def kulsinski_distance(x1, x2):
    """
    Compute the Kulsinski distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Kulsinski distance between x1 and x2.
    """
    return 1 - (np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1))

def rogers_tanimoto_distance(x1, x2):
    """
    Compute the Rogers-Tanimoto distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Rogers-Tanimoto distance between x1 and x2.
    """
    return 1 - (np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1))


def kulczynski_distance(x1, x2):
    """
    Compute the Kulczynski distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Kulczynski distance between x1 and x2.
    """
    return 1 - (np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1))


def morisita_distance(x1, x2):
    """
    Compute the Morisita distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Morisita distance between x1 and x2.
    """
    return 1 - (np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1))


def morisita_horn_distance(x1, x2):
    """
    Compute the Morisita-Horn distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Morisita-Horn distance between x1 and x2.
    """
    return 1 - (np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1))


def dice_distance(x1, x2):
    """
    Compute the Dice distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Dice distance between x1 and x2.
    """
    return 1 - (2 * np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / (np.sum(x1[:, np.newaxis, :], axis=-1) + np.sum(x2[np.newaxis, :, :], axis=-1)))


def kappa_distance(x1, x2):
    """
    Compute the Kappa distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Kappa distance between x1 and x2.
    """
    return 1 - (np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1))


def rogers_distance(x1, x2):
    """
    Compute the Rogers distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Rogers distance between x1 and x2.
    """
    return 1 - (np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1))


def jensen_distance(x1, x2):
    """
    Compute the Jensen distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Jensen distance between x1 and x2.
    """
    return 1 - (np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1))


def jensen_shannon_distance(x1, x2):
    """
    Compute the Jensen-Shannon distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Jensen-Shannon distance between x1 and x2.
    """
    return 1 - (np.sum(np.minimum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1) / np.sum(np.maximum(x1[:, np.newaxis, :], x2[np.newaxis, :, :]), axis=-1))


def hellinger_distance(x1, x2):
    """
    Compute the Hellinger distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Hellinger distance between x1 and x2.
    """
    return np.sqrt(np.sum((np.sqrt(x1[:, np.newaxis, :]) - np.sqrt(x2[np.newaxis, :, :])) ** 2, axis=-1))


def bhattacharyya_distance(x1, x2):
    """
    Compute the Bhattacharyya distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Bhattacharyya distance between x1 and x2.
    """
    return -np.log(np.sum(np.sqrt(x1[:, np.newaxis, :] * x2[np.newaxis, :, :])))


def cityblock_distance(x1, x2):
    """
    Compute the Cityblock distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Cityblock distance between x1 and x2.
    """
    return np.sum(np.abs(x1[:, np.newaxis, :] - x2[np.newaxis, :, :]), axis=-1)


def cosine_distance(x1, x2):
    """
    Compute the Cosine distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Cosine distance between x1 and x2.
    """
    return 1 - np.sum(x1[:, np.newaxis, :] * x2[np.newaxis, :, :], axis=-1) / (np.linalg.norm(x1[:, np.newaxis, :], axis=-1) * np.linalg.norm(x2[np.newaxis, :, :], axis=-1))


def correlation_distance(x1, x2):
    """
    Compute the Correlation distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.

    Returns
    -------
    distance : float
        The Correlation distance between x1 and x2.
    """
    return 1 - np.corrcoef(x1[:, np.newaxis, :], x2[np.newaxis, :, :])[0, 1]


def mahalanobis_distance(x1, x2, VI=None):
    """
    Compute the Mahalanobis distance between all pairs of x1 and x2.

    Parameters
    ----------
    x1 : array-like of shape (n_features,)
        First point.
    x2 : array-like of shape (n_features,)
        Second point.
    VI : array-like of shape (n_features, n_features), optional
        The inverse covariance matrix. If None, the covariance matrix is used.

    Returns
    -------
    distance : float
        The Mahalanobis distance between x1 and x2.
    """
    if VI is None:
        VI = np.linalg.inv(np.cov(x1, rowvar=False))
    return np.sqrt(np.dot(np.dot((x1 - x2), VI), (x1 - x2).T))
