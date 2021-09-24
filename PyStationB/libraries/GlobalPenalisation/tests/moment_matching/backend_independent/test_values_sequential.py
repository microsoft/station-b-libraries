# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Tests for the values of the simultaneous acquisition functions using the moment-matching approximation.

The tests are implemented for:
  - moment-matched q-EI
  - moment-matched q-LCB
"""
from typing import List, Literal, Tuple

import numpy as np
import pytest

import gp.moment_matching.pytorch as torch_backend
import gp.testing as testing
from gp import numeric


# --- The lists of (factories) of acquisitions to be tested. ---
# Note: Each of these is initialized in a different way. See `gp.base` to learn how the __init__ methods look like.
SEQUENTIAL_EI_ACQUISITIONS = [
    torch_backend.SequentialMomentMatchingEI,
    #  TODO: JAX & np disabled as it needs to be changed to conform to the new covariance interface
    # numpy_backend.SequentialMomentMatchingEI,
    # jax_backend.SequentialMomentMatchingEI,
]

SEQUENTIAL_LCB_ACQUISITIONS = [
    # jax_backend.SequentialMomentMatchingLCB,
]


@pytest.fixture(autouse=True)
def numpy_seed():
    np.random.seed(18920103)


def get_approximations(
    model: testing.GPyModelWrapper, selected_x: np.ndarray, candidate_x: np.ndarray
) -> List[Tuple[float, float]]:
    """

    Args:
        model: Gaussian Process regression model
        selected_x: points already selected to the batch. Shape (n_selected, input_dim).
        candidate_x: a set of candidates considered to be added to the batch. Shape (n_candidates, input_dim).

    Returns:
        for each point in `candidate_x`, the mean and standard deviation of the approximation
    """
    results = []
    for i in range(len(candidate_x)):
        xs = np.append(selected_x, candidate_x[None, i, :], axis=0)
        means, sigmas, correlations = testing.evaluate_model(model, xs)
        approximation = testing.approximate_minimum(means=means, sigmas=sigmas, correlations=correlations)
        results.append(approximation)
    return results


OneOrTwo = Literal[1, 2]


@pytest.mark.parametrize("input_dim", (2, 3, 20))
@pytest.mark.parametrize("n_training_points", (10,))
@pytest.mark.parametrize("n_selected_points", (1, 2))
@pytest.mark.parametrize("n_candidate_points", (2, 4))
@pytest.mark.parametrize("acquisition_factory", SEQUENTIAL_EI_ACQUISITIONS)
def test_sequential_ei(
    acquisition_factory, input_dim: int, n_training_points: int, n_selected_points: OneOrTwo, n_candidate_points: int
) -> None:
    # -- Mock points that have been selected to the batch and a set of candidates for the additional point
    selected_x = np.random.rand(n_selected_points, input_dim)
    candidate_x = np.random.rand(n_candidate_points, input_dim)

    # -- Initialize a model and an acquisition function
    model = testing.get_gpy_model(input_dim=input_dim, n_points=n_training_points)
    acquisition = acquisition_factory(model, selected_x=selected_x)

    # -- Calculate q-EI using the acquisition function
    qei_test = acquisition.evaluate(candidate_x)  # Shape (n_candidate_points, 1)
    assert qei_test.shape == (n_candidate_points, 1), "Shape is wrong."

    # -- Calculate the approximation considering addition of each candidate separately
    approximations = get_approximations(model=model, selected_x=selected_x, candidate_x=candidate_x)
    y_min: float = np.min(model.Y)
    qei_true_list = []
    for mean, sigma in approximations:
        qei = numeric.expected_improvement(mean=mean, standard_deviation=sigma, y_min=y_min)
        qei_true_list.append(qei)
    qei_true = np.asarray(qei_true_list).reshape((-1, 1))

    # -- Compare
    allowed_rel_error = 1e-5 if n_selected_points < 2 else 0.01  # Approximation should be exact for n_selected = 1
    assert qei_test == pytest.approx(qei_true, abs=0.0, rel=allowed_rel_error)


@pytest.mark.parametrize("input_dim", (2, 3))
@pytest.mark.parametrize("n_training_points", (10,))
@pytest.mark.parametrize("n_selected_points", (1, 2))
@pytest.mark.parametrize("n_candidate_points", (2, 4))
@pytest.mark.parametrize("acquisition_factory", SEQUENTIAL_LCB_ACQUISITIONS)
@pytest.mark.parametrize("beta", (0.3,))
def test_sequential_lcb(
    acquisition_factory,
    input_dim: int,
    n_training_points: int,
    n_selected_points: OneOrTwo,
    n_candidate_points: int,
    beta: float,
) -> None:
    # -- Mock points that have been selected to the batch and a set of candidates for the additional point
    selected_x = np.random.rand(n_selected_points, input_dim)
    candidate_x = np.random.rand(n_candidate_points, input_dim)

    # -- Initialize a model and an acquisition function
    model = testing.get_gpy_model(input_dim=input_dim, n_points=n_training_points)
    acquisition = acquisition_factory(model, selected_x=selected_x, beta=beta)

    # -- Calculate q-EI using the acquisition function
    qlcb_test = acquisition.evaluate(candidate_x)  # Shape (n_candidate_points, 1)
    assert qlcb_test.shape == (n_candidate_points, 1), "Shape is wrong."

    # -- Calculate the approximation considering addition of each candidate separately
    approximations = get_approximations(model=model, selected_x=selected_x, candidate_x=candidate_x)
    qlcb_true_list = []
    for mean, sigma in approximations:
        qlcb = numeric.lower_confidence_bound(mean=mean, standard_deviation=sigma, beta=beta)
        qlcb_true_list.append(qlcb)
    qlcb_true = np.asarray(qlcb_true_list).reshape((-1, 1))

    # -- Compare
    # TODO: The error here is larger than in qEI for n_selected_points = 2. Order in the approximation matters
    #  -- it would be good to investigate.
    ab: float = 0.01 if n_selected_points == 1 else 0.03
    assert qlcb_test == pytest.approx(qlcb_true, abs=ab, rel=0.05)
