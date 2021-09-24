# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Tests for gradients of the acquisition functions.

To test the gradients of implemented acquisition functions, append them to the lists returned by:
  - get_simultaneous_acquisitions()
  - get_sequential_acquisitions()
"""
from typing import Callable, List, Literal, Type

import numpy as np
import pytest
from scipy import optimize
from emukit.model_wrappers import GPyModelWrapper

import gp.base as base
import gp.moment_matching.pytorch as torch_backend
import gp.testing as testing


@pytest.fixture(autouse=True)
def numpy_seed():
    np.random.seed(20210334)


ORDER: Literal["C"] = "C"  # Reshaping order, used to ravel and unravel matrices. (SciPy's gradient needs 1D matrices)


def get_gmes_acquisition_factory(
    gmes_acquisition_class: Type[base.MomentMatchingGMESBase],
) -> Callable[[GPyModelWrapper], base.MomentMatchingGMESBase]:
    def gmes_constructor(model: GPyModelWrapper):
        acquisition = gmes_acquisition_class(model)
        representer_points = np.random.randn(40, model.X.shape[1])
        acquisition.update_representer_points(representer_points)
        return acquisition

    return gmes_constructor


def get_simultaneous_acquisition_classes() -> List[Callable[[GPyModelWrapper], base.SimultaneousMomentMatchingBase]]:
    """Returns *simultaneous* acquisition functions to be tested."""
    return [
        #  TODO: JAX disabled as it needs to be changed to conform to the new covariance interface
        # jax_backend.SimultaneousMomentMatchingEI,
        # functools.partial(jax_backend.SimultaneousMomentMatchingLCB, beta=0.4),
        torch_backend.SimultaneousMomentMatchingEI,
        torch_backend.SimultaneousMomentMatchingExpectedMin,
        get_gmes_acquisition_factory(torch_backend.MomentMatchingGMES),
    ]


def get_sequential_acquisition_classes() -> List[
    Callable[[GPyModelWrapper, np.ndarray], base.SimultaneousMomentMatchingBase]
]:
    """Returns *sequential* acquisition functions to be tested."""
    return [
        #  TODO: JAX disabled as it needs to be changed to conform to the new covariance interface
        # jax_backend.SequentialMomentMatchingEI,
        # functools.partial(jax_backend.SequentialMomentMatchingLCB, beta=0.1),
        torch_backend.SequentialMomentMatchingEI,
    ]


def batches_to_evaluate_at(input_dim: int, n_batches: int = 4) -> List[np.ndarray]:
    """Returns a list of batches of point candidates to evaluate the objective at.

    Args:
        input_dim: input dimension of the GP model
        n_batches: how many batches should be generated

    Returns:
        a list of length `n_batches`. The first element has shape (1, input_dim), the second (2, input_dim), and so on.
    """
    return [np.random.rand(n_points, input_dim) for n_points in range(1, n_batches + 1)]


def _calculate_step(x0: np.ndarray) -> np.ndarray:
    """Step for calculating the numerical value of the gradient at x0."""
    epsilon = 0.005
    return epsilon * x0


def err_msg(array1: np.ndarray, array2: np.ndarray) -> str:
    """Error message for array comparison."""
    return f"\n{array1}\n\t!=\n{array2}\n"


@pytest.mark.parametrize("acquisition_class", get_simultaneous_acquisition_classes())
@pytest.mark.parametrize("x0", batches_to_evaluate_at(input_dim=5))
def test_simultaneous_acquisition_gradients(
    acquisition_class: Callable[[GPyModelWrapper], base.SimultaneousMomentMatchingBase], x0: np.ndarray
) -> None:
    """Compares the returned gradient with a numeric (finite-difference) approximation.

    Note: The code of this function is complicated as scipy's check_grad requires one-dimensional arrays.

    Args:
        acquisition: simultaneous acquisition object
        x0: batch of points at which the gradient should be checked
    """
    #  - Get the acquisition
    acquisition = acquisition_class(testing.get_gpy_model(input_dim=3))
    # - Get the implemented gradient
    _, grad_f = acquisition.evaluate_with_gradients(x0)
    assert grad_f.shape == x0.shape, f"Shape of the gradient is wrong: {grad_f.shape} != {x0.shape}"

    # - Calculate the gradient numerically
    def f(x_flat: np.ndarray) -> float:
        x = x_flat.reshape(x0.shape, order=ORDER)
        return acquisition.evaluate(x)

    x0_flat = x0.ravel(order=ORDER)
    grad_f_numeric = optimize.approx_fprime(x0_flat, f, epsilon=_calculate_step(x0_flat))
    grad_f_numeric = grad_f_numeric.reshape(x0.shape, order=ORDER)

    # - Compare gradients
    np.testing.assert_allclose(grad_f, grad_f_numeric, rtol=1e-2, atol=1.5e-3, err_msg=err_msg(grad_f, grad_f_numeric))


@pytest.mark.parametrize("acquisition_class", get_sequential_acquisition_classes())
@pytest.mark.parametrize("x0", batches_to_evaluate_at(input_dim=2, n_batches=3))
@pytest.mark.parametrize("n_selected", (1, 5, 10))
def test_sequential_acquisition_gradients(
    acquisition_class: Callable[[GPyModelWrapper, np.ndarray], base.SequentialMomentMatchingBase],
    x0: np.ndarray,
    n_selected: int,
) -> None:
    # - Get the acquisition
    selected_x = np.random.rand(n_selected, 2)
    acquisition = acquisition_class(
        testing.get_gpy_model(input_dim=2),
        selected_x,
    )
    # - Get the implemented gradient
    _, grad_f = acquisition.evaluate_with_gradients(x0)  # Shape same as x0, i.e. (n_candidates, input_dim)
    assert grad_f.shape == x0.shape, f"Shape of the gradient is wrong: {grad_f.shape} != {x0.shape}"

    # - Calculate the gradient numerically, row by row
    def f(x_row: np.ndarray) -> float:
        x = x_row[None, :]  # Reshape to treat this as a one candidate point. Shape (1, input_dim)
        y = acquisition.evaluate(x)  # Array of shape (1, 1).
        return float(y)

    grad_f_numeric_list = []
    for x in x0:
        grad_f_row = optimize.approx_fprime(x, f, epsilon=_calculate_step(x))
        grad_f_numeric_list.append(grad_f_row)

    grad_f_numeric = np.stack(grad_f_numeric_list)

    # - Compare gradients
    np.testing.assert_allclose(grad_f, grad_f_numeric, rtol=1e-2, atol=1e-3, err_msg=err_msg(grad_f, grad_f_numeric))
