#!/usr/bin/env python
# Created by "Thieu" at 23:02, 02/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


def gaussian_kernel(dists, sigma):
    """
    Compute the Gaussian kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Gaussian kernel values.
    """
    return np.exp(- (dists ** 2) / (2 * sigma ** 2))


def laplace_kernel(dists, sigma):
    """
    Compute the Laplace kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Laplace kernel values.
    """
    return np.exp(- dists / sigma)


def cauchy_kernel(dists, sigma):
    """
    Compute the Cauchy kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Cauchy kernel values.
    """
    return 1. / (1. + (dists ** 2) / (sigma ** 2))


def epanechnikov_kernel(dists, sigma):
    """
    Compute the Epanechnikov kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Epanechnikov kernel values.
    """
    u = dists / sigma
    k = 0.75 * (1 - u ** 2)
    k[u > 1] = 0
    return k


def uniform_kernel(dists, sigma):
    """
    Compute the Uniform kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Uniform kernel values.
    """
    k = np.zeros_like(dists)
    k[dists <= sigma] = 1 / (2 * sigma)
    return k


def triangular_kernel(dists, sigma):
    """
    Compute the Triangular kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Triangular kernel values.
    """
    u = dists / sigma
    k = (1 - u) * (u <= 1)
    return k


def quartic_kernel(dists, sigma):
    """
    Compute the Quartic kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Quartic kernel values.
    """
    u = dists / sigma
    k = (15 / 16) * (1 - u ** 2) ** 2
    k[u > 1] = 0
    return k


def cosine_kernel(dists, sigma):
    """
    Compute the Cosine kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Cosine kernel values.
    """
    u = dists / sigma
    k = np.cos(np.pi * u / 2) * (u <= 1)
    return k


def logistic_kernel(dists, sigma):
    """
    Compute the Logistic kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Logistic kernel values.
    """
    return 1 / (1 + np.exp(dists / sigma))


def sigmoid_kernel(dists, sigma):
    """
    Compute the Sigmoid kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Sigmoid kernel values.
    """
    return np.tanh(dists / sigma)


def multiquadric_kernel(dists, sigma):
    """
    Compute the Multiquadric kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Multiquadric kernel values.
    """
    return np.sqrt(dists ** 2 + sigma ** 2)


def inverse_multiquadric_kernel(dists, sigma):
    """
    Compute the Inverse Multiquadric kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Inverse Multiquadric kernel values.
    """
    return 1 / np.sqrt(dists ** 2 + sigma ** 2)


def rational_quadratic_kernel(dists, sigma):
    """
    Compute the Rational Quadratic kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Rational Quadratic kernel values.
    """
    return 1 - (dists ** 2) / (dists ** 2 + sigma ** 2)


def exponential_kernel(dists, sigma):
    """
    Compute the Exponential kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Exponential kernel values.
    """
    return np.exp(-dists / sigma)


def power_kernel(dists, sigma):
    """
    Compute the Power kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Power kernel values.
    """
    return (1 - dists / sigma) ** 2 * (dists <= sigma)


def linear_kernel(dists, sigma):
    """
    Compute the Linear kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Linear kernel values.
    """
    return 1 - (dists / sigma)


def bessel_kernel(dists, sigma):
    """
    Compute the Bessel kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Bessel kernel values.
    """
    return np.exp(-dists / sigma) * np.cos(dists / sigma)


def vonmises_kernel(dists, sigma):
    """
    Compute the Von Mises kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Von Mises kernel values.
    """
    return np.exp(-dists / sigma) * np.sin(dists / sigma)


def vonmises_fisher_kernel(dists, sigma):
    """
    Compute the Von Mises-Fisher kernel.

    Parameters
    ----------
    dists : np.ndarray
        The distances between points.
    sigma : float
        The bandwidth parameter.

    Returns
    -------
    np.ndarray
        The computed Von Mises-Fisher kernel values.
    """
    return np.exp(-dists / sigma) * np.cos(dists / sigma)
