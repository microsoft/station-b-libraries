# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Tests for the JAX implementation of Clark's algorithm for approximation of the smallest of random variables."""
from typing import List, Tuple

import math
import numpy as np
import pytest

import gp.moment_matching.jax.clark as clark
import gp.numeric as numeric
from gp.testing import approximate_minimum

np.random.seed(42)


def compare_approximations(
    approximation_true: Tuple[float, float],
    approximation_test: Tuple[float, float],
    rel_tol: float = 0.01,
    abs_tol: float = 0.0,
) -> None:
    """A pytest-friendly check whether two tuples (mean, std) are approximately equal."""
    __tracebackhide__ = True

    mean_true, std_true = approximation_true
    mean_test, std_test = approximation_test

    if not math.isclose(mean_true, mean_test, rel_tol=rel_tol, abs_tol=abs_tol):
        pytest.fail(f"Mean is different: {mean_true:.4f} =/= {mean_test:.4f}.")

    if not math.isclose(std_true, std_test, rel_tol=rel_tol, abs_tol=abs_tol):
        pytest.fail(f"Standard deviation is different: {std_true:.4f} =/= {std_test:.4f}.")


def nondegenerate_tests_two_variables() -> List[Tuple[float, float, float, float, float]]:
    """Each entry in the list represents two Gaussian variables X1, X2 in the following format:
    (mean1, mean2, sigma1, sigma2, correlation between X1 and X2)
    """
    return [
        (1.0, 2.0, 0.5, 0.7, 0.4),
        (1.0, 3.0, 0.5, 1.0, 0.1),
        (0.0, 1.0, 10.0, 2.0, 0.3),
        (0.8, 1.0, 0.1, 0.2, 0.7),
    ]


@pytest.mark.parametrize("mu1,mu2,sigma1,sigma2,rho", nondegenerate_tests_two_variables())
def test_two_variables_nondegenerate(mu1: float, mu2: float, sigma1: float, sigma2: float, rho: float):
    means = np.array([mu1, mu2])
    sigmas = np.array([sigma1, sigma2])
    correlations = np.array([[1.0, rho], [rho, 1.0]])

    compare_approximations(
        approximate_minimum(means=means, sigmas=sigmas, correlations=correlations),
        clark.approximate_minimum(means=means, sigmas=sigmas, correlations=correlations),
    )


def degenerate_tests_two_variables() -> List[Tuple[float, float, float]]:
    """Each entry represents two Gaussian variables with the same standard deviation and that are maximally correlated:
        (mean1, mean2, sigmas).
    (As sigmas := sigma1 = sigma2 and correlation is 1).
    """
    return [
        (1.0, 2.0, 3.0),
        (2.0, 1.3, 10.0),
        (2.0, 1.2, 0.02),
    ]


@pytest.mark.parametrize("mu1,mu2,sigma", degenerate_tests_two_variables())
def test_two_variables_degenerate(mu1: float, mu2: float, sigma: float) -> None:
    means = np.array([mu1, mu2])
    sigmas = np.array([sigma, sigma])
    correlations = np.ones((2, 2), dtype=float)

    compare_approximations(
        approximate_minimum(means=means, sigmas=sigmas, correlations=correlations),
        clark.approximate_minimum(means=means, sigmas=sigmas, correlations=correlations),
    )


def three_variables_large_mean_differences() -> List[Tuple[Tuple[float, float, float], float]]:
    """Each entry represents three random variables. The format is:
    (
        (mean1, mean2, mean3),
        covariance_scale
    )
    and covariance scale controls how big the covariance between any two variables can be.

    Note:
        this data set is created in such way that differences between means are always much larger than any covariances.
        This makes the approximation very accurate.
    """
    return [
        ((1.0, 10.0, 20.0), 0.1),
        ((2.0, 9.0, 25.1), 1.0),
        ((0.1, 3.0, 5.0), 0.1),
    ]


@pytest.mark.parametrize("means,covariance_scale", three_variables_large_mean_differences())
def test_three_variables_approximation_valid(means: Tuple[float, float, float], covariance_scale: float) -> None:
    covariance = numeric.random_positive_definite_matrix(3, scale=covariance_scale)

    means = np.asarray(means).ravel()
    sigmas = numeric.covariance_into_standard_deviations(covariance)
    correlations = numeric.correlation_from_covariance(covariance)

    approximation_true = approximate_minimum(means=means, sigmas=sigmas, correlations=correlations)
    approximation_test = clark.approximate_minimum(means=means, sigmas=sigmas, correlations=correlations)

    compare_approximations(approximation_true, approximation_test)
