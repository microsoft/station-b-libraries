# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""JAX numerical utilities."""
from typing import Union

import jax
import jax.numpy as jnp

Float = Union[float, jnp.ndarray]


@jax.jit
def normal_pdf(x: Float) -> jnp.ndarray:
    """The PDF of the standard Gaussian (zero-mean, unit-variance)"""
    return jnp.exp(-0.5 * x ** 2) / jnp.sqrt(2 * jnp.pi)


@jax.jit
def normal_cdf(x: Float) -> jnp.ndarray:
    """The CDF of the standard Gaussian (zero-mean, unit-variance)"""
    return jax.scipy.special.ndtr(x)


@jax.jit
def correlation_from_covariance(covariance: jnp.ndarray) -> jnp.ndarray:
    """Normalizes covariance into correlation.

    Args:
        covariance: covariance matrix, shape (n, n)

    Returns:
        correlation: correlation matrix, shape (n, n)

    Note:
        when diagonal entries of the covariance matrix vanish, this implementation of correlation
        becomes numerically unstable
    """
    std = jnp.sqrt(jnp.diag(covariance))  # A vector of standard deviations
    outer_std = jnp.outer(std, std)  # A matrix of the form std_x * std_y
    correlation = covariance / outer_std

    correlation = jnp.clip(correlation, -1.0, 1.0)

    return correlation


@jax.jit
def expected_improvement(y_min: Float, mean: Float, standard_deviation: Float) -> Float:
    """Expected improvement.

    .. math::

        EI = \\mathbb E(max(y_min - f(x), 0))

    Args:
        y_min: observed global minimum
        mean: mean of the random variable
        standard_deviation: standard deviation of the random variable

    Returns:
        expected improvement
    """
    y_normalised: float = (y_min - mean) / standard_deviation

    return standard_deviation * (y_normalised * normal_cdf(y_normalised) + normal_pdf(y_normalised))


@jax.jit
def lower_confidence_bound(beta: Float, mean: Float, standard_deviation: Float) -> Float:
    """The *Lower* Confidence Bound acquisition function for a Gaussian variable.

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
