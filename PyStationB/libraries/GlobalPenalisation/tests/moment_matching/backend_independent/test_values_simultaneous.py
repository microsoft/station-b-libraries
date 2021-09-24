# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Tests for the values of the simultaneous acquisition functions using the moment-matching approximation.

The tests are implemented for:
  - moment-matched q-EI
  - moment-matched q-LCB
"""
from typing import Literal

import numpy as np
import pytest
import scipy.stats
from emukit.bayesian_optimization.acquisitions import MultipointExpectedImprovement

import gp.moment_matching.pytorch as torch_backend
import gp.testing as testing
from gp import numeric

# --- The lists of (factories) of acquisitions to be tested. ---
# Note: Each of these is initialized in a different way. See `gp.base` to learn how the __init__ methods look like.
SIMULTANEOUS_EI_ACQUISITIONS = [
    #  TODO: JAX disabled as it needs to be changed to conform to the new covariance interface
    # jax_backend.SimultaneousMomentMatchingEI,
    torch_backend.SimultaneousMomentMatchingEI,
]

SIMULTANEOUS_LCB_ACQUISITIONS = [
    #  TODO: JAX disabled as it needs to be changed to conform to the new covariance interface
    # jax_backend.SimultaneousMomentMatchingLCB,
]

GMES_ACQUISITION = [
    torch_backend.MomentMatchingGMES,
]


@pytest.fixture(autouse=True)
def numpy_seed():
    np.random.seed(20210401)


TwoOrThree = Literal[2, 3]


@pytest.mark.parametrize("n_candidates", (2, 3))
@pytest.mark.parametrize("input_dim", (3,))
@pytest.mark.parametrize("n_training_points", (10,))
@pytest.mark.parametrize("acquisition_factory", SIMULTANEOUS_EI_ACQUISITIONS)
def test_simultaneous_ei(acquisition_factory, n_candidates: TwoOrThree, n_training_points: int, input_dim: int) -> None:
    model = testing.get_gpy_model(input_dim=input_dim, n_points=n_training_points)
    acquisition = acquisition_factory(model=model)

    # -- Calculate the approximation of q-EI for two points using the acquisition function.
    x0 = np.random.rand(n_candidates, input_dim)
    qei_test = acquisition.evaluate(x0)

    # -- Calculate manually the approximation of q-EI.
    means, std_dev, correlation = testing.evaluate_model(model, x0)
    mu, sigma = testing.approximate_minimum(means=means, sigmas=std_dev, correlations=correlation)
    y_min = np.min(model.Y)

    qei_true = numeric.expected_improvement(mean=mu, standard_deviation=sigma, y_min=y_min)

    # -- Compare both values
    assert qei_test == pytest.approx(qei_true, rel=1e-3)


@pytest.mark.parametrize("n_candidates", [3, 5, 8])
@pytest.mark.parametrize("input_dim", [3])
@pytest.mark.parametrize("n_training_points", [5, 8])
@pytest.mark.parametrize("acquisition_factory", SIMULTANEOUS_EI_ACQUISITIONS)
def test_simultaneous_ei__approximation_close_to_true(
    acquisition_factory, n_candidates: TwoOrThree, n_training_points: int, input_dim: int
) -> None:
    num_samples = 100
    qei_test = np.zeros([num_samples])
    qei_true = np.zeros([num_samples])
    for i in range(num_samples):
        model = testing.get_gpy_model(input_dim=input_dim, n_points=n_training_points)
        acquisition = acquisition_factory(model=model)
        # -- Calculate the approximation of q-EI using the acquisition function.
        x0 = np.random.rand(n_candidates, input_dim)
        qei_test[i] = acquisition.evaluate(x0)
        # -- Calculate exact q-EI using emukit
        exact_acquisition = MultipointExpectedImprovement(model, fast_compute=False)
        qei_true[i] = -exact_acquisition.evaluate(x0)  #  Emukit acquisition returns _minus_ EI for some reason
    # Assert that estimates correlate quite highly with the true values
    assert np.corrcoef(qei_true, qei_test)[0, 1] > 0.999


@pytest.mark.parametrize("beta", (0.3, 0.7))
@pytest.mark.parametrize("input_dim", (3,))
@pytest.mark.parametrize("n_training_points", (10,))
@pytest.mark.parametrize("acquisition_factory", SIMULTANEOUS_LCB_ACQUISITIONS)
def test_simultaneous_lcb(acquisition_factory, n_training_points: int, input_dim: int, beta: float) -> None:
    model = testing.get_gpy_model(input_dim=input_dim, n_points=n_training_points)
    acquisition = acquisition_factory(model=model, beta=beta)

    # -- Calculate the approximation of q-LCB for two points using the acquisition function
    x0 = np.random.rand(2, input_dim)
    qlcb_test = acquisition.evaluate(x0)

    # -- Calculate manually the approximation of q-LCB
    means, std_dev, correlation = testing.evaluate_model(model, x0)
    mu, sigma = testing.approximate_minimum(means=means, sigmas=std_dev, correlations=correlation)

    qlcb_true = numeric.lower_confidence_bound(mean=mu, standard_deviation=sigma, beta=beta)

    # -- Compare both values
    assert qlcb_test == pytest.approx(qlcb_true, rel=1e-3)


@pytest.mark.parametrize("n_candidates", (1, 2, 3, 5))
@pytest.mark.parametrize("input_dim", (1, 2, 3))
@pytest.mark.parametrize("n_training_points", (10,))
def test_simultaneous_expected_min(n_candidates: int, n_training_points: int, input_dim: int) -> None:
    model = testing.get_gpy_model(input_dim=input_dim, n_points=n_training_points)
    acquisition = torch_backend.SimultaneousMomentMatchingExpectedMin(model=model)

    batch = np.random.rand(n_candidates, input_dim)
    # -- Calculate the acquisition
    neg_expected_min = acquisition.evaluate(batch)
    # -- Calculate Monte-carlo approximation to acquisition
    mean, cov = model.predict_with_full_covariance(batch)
    outcome_samples = scipy.stats.multivariate_normal(mean.ravel(), cov).rvs(10000)
    min_mc_samples = outcome_samples.min(axis=1) if n_candidates > 1 else outcome_samples
    expected_min_mc = min_mc_samples.mean()

    assert -expected_min_mc == pytest.approx(neg_expected_min, rel=1e-1)
