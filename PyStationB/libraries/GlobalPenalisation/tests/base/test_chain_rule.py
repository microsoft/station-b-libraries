# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Test whether the chain rule is calculated correctly.

Note:
    These tests assume that the input dimension (n_inputs) is 2.
"""
from typing import Callable, List, Tuple

import jax
import numpy as np
import pytest

from gp.base.chain_rule import (
    chain_rule_covariance,
    chain_rule_cross_covariance,
    chain_rule_means_from_predict_joint,
    chain_rule_means_vars,
)


@pytest.fixture(autouse=True)
def numpy_seed():
    np.random.seed(42)


RELATIVE_TOLERANCE: float = 1e-3


@pytest.fixture
def batch_inputs() -> np.ndarray:
    """Returns a batch od coordinates in the input space, shape (n_candidates, n_inputs).

    Recall that n_inputs = 2."""
    n_candidates = 10
    x = np.random.rand(n_candidates * 2).reshape((n_candidates, 2))

    return x


def _vectorize_gradient(f: Callable) -> Callable:
    """Having a function f, vectorise df/dx."""
    return jax.vmap(jax.grad(f))


def mean(x: Tuple[float, float]) -> float:
    """Mocks the mean value at point ``x``."""
    return 3 * x[0] ** 2 + x[1] ** 3


def mean_grad(x: Tuple[float, float]) -> Tuple[float, float]:
    """A manually calculated gradient of the mean with respect to the inputs."""
    return 6 * x[0], 3 * x[1] ** 2


def mean_vectorized(batch: np.ndarray) -> np.ndarray:
    """Vectorized version of the mean function.

    Args:
        batch: shape (n_candidates, 2)

    Returns:
        means vector, shape (n_candidates, 1)"""
    return np.asarray([[mean(x)] for x in batch])


# *** Tests of the chain rule for the sequential batch optimization ***


def test_sequential_chain_rule_means(batch_inputs) -> None:
    def acq_(mean: float) -> float:
        """Acquisition function, calculated using the mean of a Gaussian variable at x"""
        return mean ** 2

    # -- The manual chain rule, forward-mode calculation --

    # Vector of means, shape (n_candidates, 1)
    M = mean_vectorized(batch_inputs)

    # The gradient of mean with respect to input space coordinates, shape (n_candidates, n_inputs)
    dmeans_dx = np.asarray([mean_grad(x) for x in batch_inputs])

    # The derivative of acquisition with respect to the mean, shape (n_candidates, 1)
    da_dmeans = 2 * M

    # The derivative of acquisition function with respect to the input coordinates, shape (n_candidates, n_inputs)
    da_dx = chain_rule_means_vars(da_dmeans, dmeans_dx)

    # -- Let's calculate this with JAX --
    def f(x: Tuple[float, float]) -> float:
        return acq_(mean(x))

    # Shape (n_candidates, n_inputs)
    da_dx_jax = _vectorize_gradient(f)(batch_inputs)

    n_candidates: int = batch_inputs.shape[0]

    assert da_dx_jax.shape == (n_candidates, 2)
    assert da_dx.shape == (n_candidates, 2)
    np.testing.assert_allclose(da_dx, da_dx_jax, rtol=RELATIVE_TOLERANCE)


def test_sequential_chain_rule_covariance(batch_inputs) -> None:
    def cov_(x: Tuple[float, float]) -> Tuple[float, float, float]:
        """Calculates the covariances between candidate and points that already selected to the batch.
        In this case, n_selected is 3.
        """
        return 3 * x[0], x[1] ** 2, x[0] ** 2 + x[1] ** 3

    def acq_(cov: Tuple[float, float, float]) -> float:
        """Acquisition as a function of covariance vector between the candidate and the selected points."""
        return cov[0] ** 2 + cov[1] + 2 * cov[2] ** 3

    # -- The manual chain rule, forward-mode calculation --

    # Array storing covariance vectors. Shape (n_candidates, n_selected)
    cov = np.asarray([cov_(x) for x in batch_inputs])

    # Gradient of acquisition with respect to the covariance. Shape (n_candidates, n_selected)
    da_dcov = np.asarray([(2 * c[0], 1.0, 6 * c[2] ** 2) for c in cov])

    # We want the gradient of covariance with respect to the inputs. Shape
    # (n_candidates, n_selected, n_inputs). Entry (i,j,k) represents dCov(candidate_y[i], selected_y[j]) / dx[i,k]
    n_candidates = batch_inputs.shape[0]
    dcov_dx = np.zeros((n_candidates, 3, 2))

    def small_jacobian(x: Tuple[float, float]) -> np.ndarray:
        """Returns for dcov/dx for *one* candidate. Shape (n_selected, n_inputs) = (3, 2)."""
        return np.array(
            [
                [3.0, 0.0],
                [0.0, 2 * x[1]],
                [2 * x[0], 3 * x[1] ** 2],
            ]
        )

    for i, x in enumerate(batch_inputs):
        jac = small_jacobian(x)
        dcov_dx[i, :, :] = jac[:, :]

    da_dx = chain_rule_cross_covariance(da_dcov, dcov_dx)

    # -- Let's calculate this with JAX --
    def f(x: Tuple[float, float]) -> float:
        return acq_(cov_(x))

    da_dx_jax = _vectorize_gradient(f)(batch_inputs)

    assert da_dx_jax.shape == (n_candidates, 2)
    assert da_dx.shape == (n_candidates, 2)
    np.testing.assert_allclose(da_dx, da_dx_jax, rtol=RELATIVE_TOLERANCE)


# *** Tests of the chain rule for the "all at once" batch optimization ***


def test_simultaneous_chain_rule_means(batch_inputs: np.ndarray) -> None:
    def acquisition_(means: np.ndarray) -> float:
        means2 = means ** 2
        return means2.sum()

    n_candidates: int = batch_inputs.shape[0]

    # -- The manual chain rule, forward-mode calculation --

    # Vector of the means, shape (n_candidates, 1)
    M = mean_vectorized(batch_inputs)

    da_dmeans = 2 * M  # Shape (n_candidates, 1)

    # Shape (n_candidates, n_candidates, n_inputs). Initialize with 0
    dmeans_dx = np.zeros((n_candidates, n_candidates, 2))
    for i in range(n_candidates):
        dmeans_dx[i, i, :] = mean_grad(batch_inputs[i])

    da_dx = chain_rule_means_from_predict_joint(da_dmeans, dmeans_dx)

    # -- Let's calculate this with JAX --
    def f(x):
        means = jax.vmap(mean)(x)
        return acquisition_(means)

    da_dx_jax = jax.grad(f)(batch_inputs)

    assert da_dx.shape == (n_candidates, 2)
    assert da_dx_jax.shape == (n_candidates, 2)

    np.testing.assert_allclose(da_dx, da_dx_jax, rtol=RELATIVE_TOLERANCE)


def test_simultaneous_chain_rule_covariance(batch_inputs: np.ndarray) -> None:
    def covariance(xs: np.ndarray) -> List[List[float]]:
        return [[x[0] ** 2 + y[0] ** 2 + x[1] ** 3 + y[1] ** 3 for x in xs] for y in xs]

    def acquisition(cov: np.ndarray) -> float:
        return (cov ** 2).sum()

    # -- The manual chain rule, forward-mode calculation --
    n_candidates: int = batch_inputs.shape[0]

    # Covariance matrix, shape (n_candidates, n_candidates)
    cov = np.asarray(covariance(batch_inputs))

    # Derivative of acquisition with respect to the covariance matrix. Shape (n_candidates, n_candidates).
    da_dcov = 2 * cov

    # Internal check
    np.testing.assert_allclose(da_dcov, jax.grad(acquisition)(cov), rtol=RELATIVE_TOLERANCE)

    # Gradient of the covariance matrix with respect to the inputs. Shape (n_candidates, n_candidates, n_candidates, 2)
    dcov_dx = np.zeros((n_candidates, n_candidates, n_candidates, 2))

    for i, x in enumerate(batch_inputs):
        for j in range(n_candidates):
            dcov_dx[i, j, i, 0] = 2 * x[0]
            dcov_dx[j, i, i, 0] = 2 * x[0]

            dcov_dx[i, j, i, 1] = 3 * x[1] ** 2
            dcov_dx[j, i, i, 1] = 3 * x[1] ** 2

    # Now set the diagonal terms
    for i, x in enumerate(batch_inputs):
        dcov_dx[i, i, i, 0] = 2 * 2 * x[0]
        dcov_dx[i, i, i, 1] = 2 * 3 * x[1] ** 2

    # Internal check of this large matrix with JAX
    def covariance_jax_wrapper(xs):
        return jax.numpy.asarray(covariance(xs))

    dcov_dx_jax = jax.jacfwd(covariance_jax_wrapper)(batch_inputs)
    np.testing.assert_allclose(dcov_dx, dcov_dx_jax, rtol=RELATIVE_TOLERANCE)

    # Use chain rule
    da_dx = chain_rule_covariance(da_dcov, dcov_dx)

    # -- Let's calculate this with JAX --
    def f(x):
        cov = jax.numpy.asarray(covariance(x))
        return acquisition(cov)

    da_dx_jax = jax.grad(f)(batch_inputs)

    assert da_dx_jax.shape == (n_candidates, 2)
    assert da_dx.shape == (n_candidates, 2)
    np.testing.assert_allclose(da_dx, da_dx_jax, rtol=RELATIVE_TOLERANCE)
