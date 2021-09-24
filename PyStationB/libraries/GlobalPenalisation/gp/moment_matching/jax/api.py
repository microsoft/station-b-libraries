# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Exports:
    SequentialMomentMatchingEI
    SequentialMomentMatchingUCB
    SimultaneousMomentMatchingEI
    SimultaneousMomentMatchingLCB
"""
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.ops as ops

import numpy as np

import gp.base as base
import gp.moment_matching.jax.clark as clark
import gp.moment_matching.jax.numeric as numeric


@jax.jit
def _append_candidate_correlation(
    selected_correlation: jnp.ndarray, candidate_to_selected_correlation: jnp.ndarray
) -> jnp.ndarray:
    """Extends a correlation matrix by a new variable.

    Args:
        selected_correlation: correlation matrix between variables Y1, ..., Yn, shape (n, n)
        candidate_to_selected_correlation: correlations between a variable Z and Y1, ..., Yn. Shape (n,)

    Returns:
        correlation matrix for variables Y1, ..., Yn, Z. Shape (n+1, n+1).
    """
    n = selected_correlation.shape[0]
    new_correlations = jnp.zeros((n + 1, n + 1))

    # Add a column and a row with the correlations
    new_correlations = ops.index_update(new_correlations, ops.index[n, 0:n], candidate_to_selected_correlation)
    new_correlations = ops.index_update(new_correlations, ops.index[0:n, n], candidate_to_selected_correlation)

    # Correlation between Z to itself is 1.0
    new_correlations = ops.index_update(new_correlations, ops.index[n, n], 1.0)

    return new_correlations


@jax.jit
def _append_candidate_value(selected_values: jnp.ndarray, candidate_value: float) -> jnp.ndarray:
    """Extends a vector with values by another value.

    Args:
        selected_values: means/standard deviations vector for Y1, ..., Yn. Shape (n,), (1,n) or (n,1).
        candidate_value: mean/standard deviation for Z

    Returns:
        values for Y1, ..., Yn, Z. Shape (n+1,)
    """
    reshaped_input = jnp.reshape(selected_values, (-1,))

    n = selected_values.shape[0]
    new_vector = jnp.zeros((n + 1,))

    new_vector = ops.index_update(new_vector, ops.index[:n], reshaped_input)
    new_vector = ops.index_update(new_vector, ops.index[n], candidate_value)

    return new_vector


@jax.jit
def _correlation_vector(covariance: jnp.ndarray, selected_std: jnp.ndarray, candidate_std: float) -> jnp.ndarray:
    """Rescales covariances into vectors.

    Args:
        selected_std: standard deviations of selected points, shape (n, 1) or (n,).
        candidate_std: standard deviation of the candidate.
        covariance: covariance vector, shape (n, 1) or (n,)

    Returns:
        correlations vector. Shape (n,).
    """
    return covariance.ravel() / (selected_std.ravel() * candidate_std)


def _sequential_moment_matching_approximation(
    candidate_mean: float,
    candidate_std: float,
    candidate_to_selected_covariance: jnp.ndarray,
    selected_mean: jnp.ndarray,
    selected_std: jnp.ndarray,
    selected_correlation: jnp.ndarray,
) -> Tuple[float, float]:
    """Consider Y1, ..., Yn variables corresponding to the selected points in the batch and a candidate for the (n+1)th
    point Z.

    This function calculates the moment-matching approximation of min(Y1, ..., Yn, Z).

    Args:
        candidate_mean: mean of Z
        candidate_std: standard deviation of Z
        candidate_to_selected_covariance: vector of correlations from Z to Y1, ..., Yn. Shape (n, 1), (1, n) or (n,).
        selected_mean: means of Y1, ..., Yn. Shape (n, 1), (1, n) or (n,)
        selected_std: standard deviations of Y1, ..., Yn. Shape as above
        selected_correlation: correlation matrix for Y1, ..., Yn. Shape (n, n).

    Returns:
        mean of min(Y1, ..., Yn, Z)
        standard deviation of min(Y1, ..., Yn, Z)
    """
    correlation_vector = _correlation_vector(
        candidate_to_selected_covariance, selected_std=selected_std, candidate_std=candidate_std
    )

    correlation_matrix = _append_candidate_correlation(
        selected_correlation=selected_correlation, candidate_to_selected_correlation=correlation_vector
    )

    means = _append_candidate_value(selected_mean, candidate_mean)
    stds = _append_candidate_value(selected_std, candidate_std)

    return clark.approximate_minimum(means=means, sigmas=stds, correlations=correlation_matrix)


class SequentialMomentMatchingEI(base.SequentialMomentMatchingBase):
    def _evaluate_with_gradients(
        self, y_mean: np.ndarray, y_std: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Get the minimum as observed so far
        y_min: float = self.model.Y.min()

        def f(y_mean: float, y_std: float, covariance: jnp.ndarray) -> float:
            mu, sigma = _sequential_moment_matching_approximation(
                candidate_mean=y_mean,
                candidate_std=y_std,
                candidate_to_selected_covariance=covariance,
                selected_mean=self.selected_y_mean,
                selected_std=self.selected_y_std,
                selected_correlation=self.selected_y_correlation,
            )
            return numeric.expected_improvement(y_min=y_min, mean=mu, standard_deviation=sigma)

        # Vectorize f
        vect_f = jax.vmap(jax.value_and_grad(f, argnums=(0, 1, 2)))
        values, (grad1, grad2, grad3) = vect_f(y_mean.ravel(), y_std.ravel(), covariance)
        return np.asarray(values)[:, None], np.asarray(grad1)[:, None], np.asarray(grad2)[:, None], np.stack(grad3)


class SequentialMomentMatchingLCB(base.SequentialMomentMatchingBaseLCB):
    def _evaluate_with_gradients(
        self, y_mean: np.ndarray, y_std: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        def f(y_mean: float, y_std: float, covariance: jnp.ndarray) -> float:
            mu, sigma = _sequential_moment_matching_approximation(
                candidate_mean=y_mean,
                candidate_std=y_std,
                candidate_to_selected_covariance=covariance,
                selected_mean=self.selected_y_mean,
                selected_std=self.selected_y_std,
                selected_correlation=self.selected_y_correlation,
            )
            return numeric.lower_confidence_bound(beta=self.beta, mean=mu, standard_deviation=sigma)

        # Vectorize f
        vect_f = jax.vmap(jax.value_and_grad(f, argnums=(0, 1, 2)))
        values, (grad1, grad2, grad3) = vect_f(y_mean.ravel(), y_std.ravel(), covariance)
        return np.asarray(values)[:, None], np.asarray(grad1)[:, None], np.asarray(grad2)[:, None], np.stack(grad3)


@jax.jit
def _moment_matching_from_covariance(means: jnp.ndarray, covariance: jnp.ndarray) -> Tuple[float, float]:
    """Mean and standard deviation of the minimum of a set of random variables.

    Args:
        means: means of X1, ..., Xn. Shape (n, 1) or (n,).
        covariance: covariance matrix of X1, ..., Xn. Shape (n, n).

    Returns:
        mean of min(X1, ..., Xn)
        standard deviation of min(X1, ..., Xn)
    """
    means = means.ravel()
    sigmas = jnp.sqrt(jnp.diag(covariance))
    correlations = numeric.correlation_from_covariance(covariance)
    mean, sigma = clark.approximate_minimum(means=means, sigmas=sigmas, correlations=correlations)

    return mean, sigma


class SimultaneousMomentMatchingEI(base.SimultaneousMomentMatchingBase):
    def _evaluate_with_gradients(
        self, y_mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:

        # TODO: Is this the right way to retrieve that?
        y_min: float = self.model.Y.min()

        def f(y_mean: jnp.ndarray, covariance: jnp.ndarray) -> float:
            mean, sigma = _moment_matching_from_covariance(y_mean, covariance)
            return numeric.expected_improvement(y_min=y_min, mean=mean, standard_deviation=sigma)

        # Return the value and the gradients
        value, (grad_means, grad_covariance) = jax.value_and_grad(f, argnums=(0, 1))(y_mean, covariance)
        return float(value), np.asarray(grad_means), np.asarray(grad_covariance)


class SimultaneousMomentMatchingLCB(base.SimultaneousMomentMatchingBaseLCB):
    def _evaluate_with_gradients(
        self, y_mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:

        # Create an auxilary function, calculating the acquisition function using JAX backend
        def f(y_mean: jnp.ndarray, covariance: jnp.ndarray) -> float:
            mean, sigma = _moment_matching_from_covariance(y_mean, covariance)

            return numeric.lower_confidence_bound(beta=self.beta, mean=mean, standard_deviation=sigma)

        # Return the value and the gradients
        value, (grad_means, grad_covariance) = jax.value_and_grad(f, argnums=(0, 1))(y_mean, covariance)
        return float(value), np.asarray(grad_means), np.asarray(grad_covariance)
