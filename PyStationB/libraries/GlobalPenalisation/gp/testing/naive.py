# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This is a *very* simple Python implementation of the Clark's algorithm for two and three variables. It is intended
for testing purposes only.
"""
from typing import Tuple

import numpy as np
import scipy.stats as stats


def _degenerate_check(a: float) -> bool:
    """True iff a is approximately 0."""
    return a < 1e-9


def pdf(x: float) -> float:
    return stats.norm.pdf(x)


def cdf(x: float) -> float:
    return stats.norm.cdf(x)


def calculate_a(sigma1: float, sigma2: float, rho: float) -> float:
    """Calculates Clark's expression for a (degeneracy condition)."""
    a2 = sigma1 ** 2 + sigma2 ** 2 - 2 * sigma1 * sigma2 * rho
    a = a2 ** 0.5
    return a


def approximate_minimum_two(mu1: float, mu2: float, sigma1: float, sigma2: float, rho: float) -> Tuple[float, float]:
    """Consider Gaussian variables X1 and X2. This formula approximates min(X1, X2) by a Gaussian variable.

    Args:
        mu1: mean of X1
        mu2: mean of X2
        sigma1: standard deviation of X1
        sigma2: standard deviation of X2
        rho: correlation coefficient between X1 and X2

    Returns:
        mean of the approximation to min(X1, X2)
        standard deviation of the approximation to min(X1, X2)
    """
    a = calculate_a(sigma1, sigma2, rho)

    if _degenerate_check(a):
        return min(mu1, mu2), sigma1

    else:
        alpha = (mu2 - mu1) / a
        mu = mu1 * cdf(alpha) + mu2 * cdf(-alpha) - a * pdf(alpha)
        nu2 = (
            (mu1 ** 2 + sigma1 ** 2) * cdf(alpha)
            + (mu2 ** 2 + sigma2 ** 2) * cdf(-alpha)
            - (mu1 + mu2) * a * pdf(alpha)
        )

        return mu, (nu2 - mu ** 2) ** 0.5


def approximate_minimum_three(
    mu1: float,
    mu2: float,
    mu3: float,
    sigma1: float,
    sigma2: float,
    sigma3: float,
    rho12: float,
    rho23: float,
    rho13: float,
) -> Tuple[float, float]:
    """Approximates min(X1, X2, X3) as min( min(X1, X2), X3 ).

    Args:
        mu1: mean of X1
        mu2: mean of X2
        mu3: mean of X3
        sigma1: standard deviation of X1
        sigma2: standard deviation of X2
        sigma3: standard deviation of X3
        rho12: correlation coefficient between X1 and X2
        rho23: correlation coefficient between X2 and X3
        rho13: correlation coefficient between X1 and X3

    Returns:
        mean of an approximation to min(X1, X2, X3)
        standard deviation of an approximation to min(X1, X2, X3)

    Note:
        The approximation depends on the order of the variables!
    """
    mu_glued, sigma_glued = approximate_minimum_two(mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2, rho=rho12)
    a = calculate_a(sigma1, sigma2, rho12)

    if _degenerate_check(a):  # In this case, necessarily rho23 = rho13
        rho_glued3 = rho23
    else:
        alpha = (mu2 - mu1) / a
        rho_glued3 = (sigma1 * rho13 * cdf(alpha) + sigma2 * rho23 * cdf(-alpha)) / sigma_glued

    return approximate_minimum_two(mu1=mu_glued, mu2=mu3, sigma1=sigma_glued, sigma2=sigma3, rho=rho_glued3)


def approximate_minimum(means: np.ndarray, sigmas: np.ndarray, correlations: np.ndarray) -> Tuple[float, float]:
    """Approximates min(X1, ..., Xn) for n = 2 or n = 3.

    Args:
        means: mean of X1, ..., Xn
        sigmas: standard deviation of X1, ..., Xn
        correlations: correlations between Xi and Xj

    Returns:
        approximate mean of min(X1, ..., Xn)
        approximate standard deviation of min(X1, ..., Xn)
    """
    n = means.shape[0]
    assert means.shape == (n,) and sigmas.shape == (n,) and correlations.shape == (n, n), "Shapes don't match."

    if n == 2:
        return approximate_minimum_two(
            mu1=means[0], mu2=means[1], sigma1=sigmas[0], sigma2=sigmas[1], rho=correlations[0, 1]
        )
    elif n == 3:
        return approximate_minimum_three(
            mu1=means[0],
            mu2=means[1],
            mu3=means[2],
            sigma1=sigmas[0],
            sigma2=sigmas[1],
            sigma3=sigmas[2],
            rho13=correlations[0, 2],
            rho23=correlations[1, 2],
            rho12=correlations[0, 1],
        )
    else:
        raise ValueError(f"Number of variables must be 2 or 3. (Was {n}).")
