# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""A JAX implementation of Clark's moment-matching algorithm.

Exports:
    approximate_minimum, the Gaussian approximate of the minimum of a set of random variables

"""
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.ops as ops

import gp.moment_matching.jax.numeric as numeric


def calculate_a(sigma1: float, sigma2: float, rho: float) -> float:
    """
    Args:
        sigma1: standard deviation of the first Gaussian random variable :math:`Y_1`
        sigma2: standard deviation of the second Gaussian random variable :math:`Y_2`
        rho: the coefficient of linear correlation :math:`r(Y_1, Y_2)`
    """
    return jnp.sqrt(sigma1 ** 2 + sigma2 ** 2 - 2 * sigma1 * sigma2 * rho)


def calculate_alpha(mu1: float, mu2: float, a: float) -> float:
    """
    Args:
        mu1: mean of the first Gaussian random variable :math:`Y_1`
        mu2: mean of the second Gaussian random variable :math:`Y_2`
        a: this number is calculated using standard deviations and the coefficent of linear correlation

    Note:
        alpha can't be calculated in the case :math:`a=0`.
    """
    return (mu2 - mu1) / a


def calculate_first_moment_degenerate(mu1: float, mu2: float) -> float:
    return jnp.minimum(mu1, mu2)


def calculate_first_moment_nondegenerate(mu1: float, mu2: float, a: float, alpha: float) -> float:
    """The first moment (mean) of the random variable min(X1, X2).

    Args:
        mu1: mean of X1
        mu2: mean of X2
        a: value of the a function for X1 and X2
        alpha: value of the alpha function for X1 and X2

    Note:
        This expression assumes that ``a`` is non-zero
    """
    return mu1 * numeric.normal_cdf(alpha) + mu2 * numeric.normal_cdf(-alpha) - a * numeric.normal_pdf(alpha)


def calculate_second_moment_degenerate(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    return jnp.where(mu1 < mu2, mu1 ** 2 + sigma1 ** 2, mu2 ** 2 + sigma2 ** 2)


def calculate_second_moment_nondegenerate(
    mu1: float, mu2: float, sigma1: float, sigma2: float, a: float, alpha: float
) -> float:
    """The second (raw) moment of a random variable :math:`\\min(Y_1, Y_2)`.

    Args:
        mu1: mean of the first Gaussian random variable :math:`Y_1`
        mu2: mean of the second Gaussian random variable :math:`Y_2`
        sigma1: standard deviation of the first Gaussian random variable :math:`Y_1`
        sigma2: standard deviation of the second Gaussian random variable :math:`Y_2`
        a: value of a(X1, X2)
        alpha: value of alpha(X1, X2)

    Note:
        For a Gaussian variable, the relationship between the raw second moment, mean, and the standard deviation
        (which is calculated using the *central* moment) is

        .. math::

            \\nu_2 = \\nu_1^2 + \\sigma^2
    """
    # The first, second and third term
    first = (mu1 ** 2 + sigma1 ** 2) * numeric.normal_cdf(alpha)
    secnd = (mu2 ** 2 + sigma2 ** 2) * numeric.normal_cdf(-alpha)
    third = (mu1 + mu2) * a * numeric.normal_pdf(alpha)

    return first + secnd - third


def standard_deviation_from_moments(nu1: float, nu2: float) -> float:
    return jnp.sqrt(nu2 - nu1 ** 2)


def calculate_correlations_vector(
    alpha: float, sigma_min: float, sigma1: float, sigma2: float, rho1: jnp.ndarray, rho2: jnp.ndarray
) -> jnp.ndarray:
    """Calculates correlation between min(X1, X2) and a batch of other variables Y1, ..., Yn.

    Args:
        alpha: alpha for X1 and X2
        sigma_min: standard deviation of min(X1, X2)
        sigma1: standard deviation of X1
        sigma2: standard deviation of X2
        rho1: vector of correlations between X1 and variables Y1, Y2, ..., Yn. Shape (n,)
        rho2: vector of correlations between X2 and variables Y1, Y2, ..., Yn. Shape (n,)

    Returns:
        vector of correlations between min(X1, X2) and variables Y1, Y2, ..., Yn. Shape (n,)
    """
    numerator = sigma1 * rho1 * numeric.normal_cdf(alpha) + sigma2 * rho2 * numeric.normal_cdf(-alpha)

    return numerator / sigma_min


def _degenerate_check(a: float) -> bool:
    return a == 0


def _degenerate_rewrite(
    mean1: float, mean2: float, sigma1: float, sigma2: float, correlations: jnp.ndarray, a: float = 0.0
) -> Tuple[float, float, jnp.ndarray]:
    """
    Args:
        mean1: mean of X1
        mean2: mean of X2
        sigma1: standard deviation of X1
        sigma2: standard deviation of X2
        correlations: array of correlations, shape (n, n)
        a: in this case a = 0, but we want to match the API of

    Returns:
        mean of min(X1, X2)
        standard deviation of min(X1, X2)
        correlations between min(X1, X2) and X1, X2, ..., Xn. Shape (n,)

    Note:
        Argument `a` is not used, but we need to match the signature of the `_nondegenerate_rewrite()` method.
    """
    mean_new = calculate_first_moment_degenerate(mu1=mean1, mu2=mean2)
    second_moment = calculate_second_moment_degenerate(mu1=mean1, sigma1=sigma1, mu2=mean2, sigma2=sigma2)
    sigma_new = standard_deviation_from_moments(nu1=mean_new, nu2=second_moment)

    correlations_vector = correlations[0]

    return mean_new, sigma_new, correlations_vector


def _nondegenerate_rewrite(
    mean1: float, mean2: float, sigma1: float, sigma2: float, a: float, correlations: jnp.ndarray
) -> Tuple[float, float, jnp.ndarray]:
    """
    Args:
        mean1: mean of X1
        mean2: mean of X2
        sigma1: standard deviation of X1
        sigma2: standard deviation of X2
        correlations: array of correlations, shape (n, n)

    Returns:
        mean of min(X1, X2)
        standard deviation of min(X1, X2)
        correlations between min(X1, X2) and X1, X2, ..., Xn. Shape (n,)
    """
    alpha = calculate_alpha(mu1=mean1, mu2=mean2, a=a)
    mean_new = calculate_first_moment_nondegenerate(mu1=mean1, mu2=mean2, a=a, alpha=alpha)
    second_moment = calculate_second_moment_nondegenerate(
        mu1=mean1, mu2=mean2, sigma1=sigma1, sigma2=sigma2, a=a, alpha=alpha
    )
    sigma_new = standard_deviation_from_moments(nu1=mean_new, nu2=second_moment)

    rho1, rho2 = correlations[0], correlations[1]
    correlations_to_new = calculate_correlations_vector(
        alpha=alpha, sigma_min=sigma_new, sigma1=sigma1, sigma2=sigma2, rho1=rho1, rho2=rho2
    )

    correlations_to_new = jnp.clip(correlations_to_new, -1.0, 1.0)

    return mean_new, sigma_new, correlations_to_new


def _degenerate_wrapper(operand: dict) -> Tuple[float, float, jnp.ndarray]:
    return _degenerate_rewrite(**operand)


def _nondegenerate_wrapper(operand: dict) -> Tuple[float, float, jnp.ndarray]:
    return _nondegenerate_rewrite(**operand)


def _update_vector(old_vector: jnp.ndarray, value: float) -> jnp.ndarray:
    """Let ``f`` be a function of a random variable (e.g. a mean or standard deviation)

    Args:
        old_vector: a vector f(X1), f(X2), ... f(Xn). Shape (n,)
        value: the value of f(Y), where Y = min(X1, X2)

    Returns:
        a vector f(Y), f(X3), ..., f(Xn). Shape (n-1,)
    """
    return ops.index_update(old_vector[1:], ops.index[0], value)


def _update_correlations(old_correlations: jnp.ndarray, correlations_vector: jnp.ndarray) -> jnp.ndarray:
    """

    Args:
        old_correlations: correlation matrix for X1, X2, ..., Xn. Shape (n, n).
        correlations_vector: correlations vector between min(X1, X2) and X1, X2, ..., Xn

    Returns:
        correlations matrix for variables min(X1, X2), X3, ..., Xn. Shape (n-1, n-1)

    Note:
        first two entries of ``correlations_vector`` are not used. They are just for a convenient API
    """
    # Correlations Xi, Xj are not changed for i, j >= 3. Copy these entries
    new_correlations = old_correlations[1:, 1:]  # Shape (n-1, n-1)

    # Vector of correlations for min(X1, X2) and X3, ..., Xn. Shape (n-2)
    correlations_to_min = correlations_vector[2:]

    # Update the entries in the first row and first column with the recalculated entries
    new_correlations = ops.index_update(new_correlations, ops.index[0, 1:], correlations_to_min)
    new_correlations = ops.index_update(new_correlations, ops.index[1:, 0], correlations_to_min)

    return new_correlations


@jax.jit
def approximate_minimum(means: jnp.ndarray, sigmas: jnp.ndarray, correlations: jnp.ndarray) -> Tuple[float, float]:
    """Consider a set of Gaussian random variables X1, ..., Xn.

    We will recursively apply the formulae above to estimate the mean and standard deviation of min(X1, ..., Xn)

    Args:
        means: means of random variables. Shape (n,)
        sigmas: standard deviations of random variables. Shape (n,)
        correlations: correlation matrix. Shape (n, n)

    Returns:
        approximation of the mean of min(Y1, ..., Yn)
        approximation of the standard deviation of min(Y1, ..., Yn)

    Note:
        this function is not JIT-compatible
    """
    # This is the trivial case
    if len(means) == 1:
        return means[0], sigmas[0]

    # Otherwise, we have two cases.
    sigma1, sigma2 = sigmas[0], sigmas[1]
    rho = correlations[0][1]

    a = calculate_a(sigma1=sigma1, sigma2=sigma2, rho=rho)
    mu1, mu2 = means[0], means[1]

    # Check if we have the degenerate case.
    # I would prefer not to use an if statement instead of jax.lax.cond, but it's not compatible with jit and vmap
    mean_new, sigma_new, correlations_new = jax.lax.cond(
        _degenerate_check(a),
        _degenerate_wrapper,
        _nondegenerate_wrapper,
        operand={
            "mean1": mu1,
            "mean2": mu2,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "correlations": correlations,
            "a": a,
        },
    )

    # If we have only two variables, that's it -- we can finish
    if len(means) == 2:
        return mean_new, sigma_new

    # Otherwise, we need to run this function with n-1 variables in total: min(X1, X2), X3, ..., Xn.
    # Replace X1 and X2 with min(X1, X2). Hence, the shapes of the following vectors are (n-1,)
    updated_means = _update_vector(means, mean_new)
    updated_sigmas = _update_vector(sigmas, sigma_new)

    # The shape of the following matrix is (n-1, n-1)
    updated_correlations = _update_correlations(correlations, correlations_new)

    return approximate_minimum(updated_means, updated_sigmas, updated_correlations)
