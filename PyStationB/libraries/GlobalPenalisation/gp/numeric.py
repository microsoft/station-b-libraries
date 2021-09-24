# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Utility functions for matrix/numerical operations. This should not depend on Emukit/GPy/JAX/pytorch -- it's bare
NumPy and maths.

Exports:
    correlation_from_covariance: passing from a covariance matrix to the correlation matrix
    batch_correlation_from_covariance: a vectorized version of correlation_from_covariance
    sigmoid: the sigmoid function
"""
from typing import Optional, Tuple

import numpy as np
import scipy.stats


def standardize(x: np.ndarray) -> np.ndarray:
    """
    Standardize an array by subtracting the mean of its elements, and deviating by std of its
    elements.
    """
    return (x - x.mean()) / x.std()


def covariance_into_standard_deviations(covariance: np.ndarray) -> np.ndarray:
    """

    Args:
        covariance: covariance matrix between variables Y1, ..., Yn. Shape (..., n, n).

    Returns:
        standard deviations of variables Y1, ..., Yn. Shape (..., n).
    """
    std = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    return std


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Correlation matrix from covariance matrix.

    Adapted from https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b

    Args:
        covariance: covariance matrix, shape (n, n)

    Returns:
        correlation matrix, shape (n, n)

    See:
        For a batch of covariance matrices, consider ``batch_correlation_from_covariance``.
    """
    std = np.sqrt(np.diag(covariance))  # A vector of standard deviations
    outer_std = np.outer(std, std)  # A matrix of the form std_x * std_y
    correlation = covariance / outer_std
    correlation[covariance == 0] = 0  # Mask fields where correlation is not defined.
    return correlation


def correlation_between_distinct_sets_from_covariance(
    covariance: np.ndarray, std1: np.ndarray, std2: np.ndarray
) -> np.ndarray:
    """Correlation matrix between two sets of random variables from the corresponding covariance matrix.

    Let's say the first set has (n1) variables, and the second has (n2) variables.

    Note: (...) indicates this function allows for prepending batch dimensions (and numpy broadcasting)

    Args:
        covariance: covariance matrix, shape (..., n1, n2)
        std1: standard deviations of the first set of random variables of shape (..., n1)
        std2: standard deviations of the second set of random variables of shape (..., n2)

    Returns:
        correlation matrix, shape (..., n1, n2) (same shape as covariance)

    See:
        For a batch of covariance matrices, consider ``batch_correlation_from_covariance``.
    """
    outer_std = std1[..., None] * std2[..., None, :]  # A matrix of the form std_x * std_y
    correlation = covariance / outer_std
    correlation[covariance == 0] = 0  # Mask fields where correlation is not defined.
    return correlation


def sigmoid(x: np.ndarray) -> np.ndarray:
    # TODO: Maybe we should consider a numerically stable alternative: https://stackoverflow.com/q/51976461
    #  For now it's only used for test functions, so it's never in the numerically unstable regime
    return np.exp(x) / (1 + np.exp(x))


def batch_correlation_from_covariance(covariance: np.ndarray, standard_deviations: Optional[np.ndarray]) -> np.ndarray:
    """Compute the correlation matrices from an array of covariance matrices.

    Args:
        covariance (np.ndarray): Array of shape (num_cases, num_dim, num_dim) where (for every i)
            the covariance[i, :, :] matrix is a positive semi-definite covariance matrix for a multivariate Gaussian.
        standard_deviations: A (num_cases, num_dim) shaped array of standard deviations corresponding to the
            covariance matrices. This argument is optional if standard deviations have been precomputed before calling.

    Returns:
        np.ndarray: A (num_cases, num_dim, num_dim) shaped array of correlation matrices.

    See:
        For a single covariance matrix, consider using ``correlation_from_covariance``.
    """
    # stds_outer_product = standard_deviations[:, :, None] * standard_deviations[:, None, :]
    stds_outer_product = np.einsum("...i,...j->...ij", standard_deviations, standard_deviations)
    correlation = covariance / stds_outer_product
    correlation[covariance == 0] = 0
    # Due to numerical errors, sometimes abs(correlation) > 1.0 by a narrow margin
    correlation = np.clip(correlation, -1.0, 1.0)
    return correlation


def expected_improvement(y_min: np.ndarray, mean: np.ndarray, standard_deviation: np.ndarray) -> np.ndarray:
    """Expected improvement.

    .. math::

        EI = \\mathbb E(max(y_min - f(x), 0))

    Args:
        y_min: observed global minimum
        mean: mean of the random variable
        standard_deviation: standard deviation of the random variable

    Returns:
        Expected improvement
    """
    y_normalised = (y_min - mean) / standard_deviation
    pdf = scipy.stats.norm.pdf(y_normalised)
    cdf = scipy.stats.norm.cdf(y_normalised)

    return standard_deviation * (y_normalised * cdf + pdf)


def random_positive_definite_matrix(size: int, scale: float = 1.0) -> np.ndarray:
    """A random symmetric positive definite matrix of given size.

    Args:
        size: dimension of the generated matrix
        scale: parameter by which the entries are scaled

    Returns:
        a symmetric positive definite matrix of shape (size, size)
    """
    factor = np.random.rand(size, size)
    mat = factor.T @ factor
    return scale * mat


def lower_confidence_bound(beta: float, mean: np.ndarray, standard_deviation: np.ndarray) -> np.ndarray:
    """The *Lower* Confidence Bound acquisition function.

    .. math::

        UCB = \\mu - \\beta \\cdot \\sigma

    Args:
        beta: the parameter of UCB
        mean: the mean of normal variable
        standard_deviation: standard deviation of the normal variable

    Returns:
        the Lower Confidence Bound
    """
    return mean - beta * standard_deviation


def generate_random_variables(
    n_batch: int, size: int, mean_loc: float = 0, mean_scale: float = 1, covariance_scale: float = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates a batch of sets of random variables.

    Args:
        n_batch: length of the batch generated
        size: the number of random variables
        mean_loc: controls how the entries for the means matrix are generated
        mean_scale: controls how the entries for the means matrix are generated
        covariance_scale: controls how the entries of the covariance matrices are generated

    Returns:
        means: means of the random variables, shape (n_batch, size)
        covariance: array of shape (n_batch, size, size), storing the covariance matrix for variables
            in each set in the batch. Each covariance matrix has entries from the range [0, 1]
        stds: array of standard deviations, shape (n_batch, size)
        correlation: array of correlations between variables in each set, shape (n_batch, size, size)
    """
    means = np.random.normal(size=(n_batch, size), loc=mean_loc, scale=mean_scale)
    covariance = np.stack(
        [random_positive_definite_matrix(size, scale=covariance_scale) for _ in range(n_batch)], axis=0
    )

    stds = np.sqrt(np.einsum("ijj->ij", covariance))
    correlation = batch_correlation_from_covariance(covariance, stds)

    return means, covariance, stds, correlation


def monte_carlo_minimum_estimate(
    means: np.ndarray, covariances: np.ndarray, n_samples: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized function that yields the estimate of mean and standard deviation of the smallest of a set of
        random variables. In other words, we consider a ``n_batch`` sets of ``n_variables`` variables and we want
        ``n_batch`` estimates of means and ``n_batch`` estimates of standard deviation.

    Args:
        means: the means of the batch of sets of random variables. Shape (n_batch, n_variables)
        covariances: the batch of covariance matrices. Shape (n_batch, n_variables, n_variables)
        n_samples: number of Monte Carlo samples to be drawn for each set in the batch

    Returns:
        means_of_minimum: a vector of shape (n_batch,) with the means of the minimum of each set of random variables
        stds_of_minimum: a vector of  shape (n_batch,) with the standard deviations of the minimum of each set of
            random variables

    Raises:
        ValueError, if the shape of means or covariance matrices is wrong
    """
    n_batch: int = means.shape[0]
    size: int = means.shape[1]

    if means.shape != (n_batch, size):
        raise ValueError("The batch of mean vectors has a wrong shape.")
    if covariances.shape != (n_batch, size, size):
        raise ValueError("The batch of covariance matrices has a wrong shape.")

    mc_means, mc_stds = np.zeros(n_batch), np.zeros(n_batch)
    for i in range(n_batch):
        # Shape (n_samples, n_variables).
        samples_variables = np.random.multivariate_normal(mean=means[i], cov=covariances[i], size=n_samples)

        # Shape (n_samples,). These are samples from the distribution of min(Y1, ..., Yn)
        samples_min = np.min(samples_variables, axis=1)

        # As we have samples from min(Y1, ..., Yn), we can calculate the mean and standard deviation
        mc_means[i] = samples_min.mean()
        mc_stds[i] = samples_min.std()

    return mc_means, mc_stds
