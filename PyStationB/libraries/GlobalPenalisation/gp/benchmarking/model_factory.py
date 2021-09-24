# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Gaussian Process model factory methods."""
import gpytorch
import GPy
import numpy as np
import torch

from emukit.core import ParameterSpace
from emukit.model_wrappers import GPyModelWrapper

import gp.model_wrapper as mw
from gp.model_wrapper import AxisReflectiveSymmetricKernel
from gp.benchmarking.benchmark_functions import BenchmarkFunctions
from gp.gpy.kernels_with_grad import Add, Symmetric

NOISE_LEVEL: float = 1e-4  # Note: too small value and the algorithms will be unstable


def get_model_for_scenario_gpy(
    X_init: np.ndarray, Y_init: np.ndarray, scenario: BenchmarkFunctions, n_restarts: int = 5
) -> GPyModelWrapper:
    """Get a model from initial data points for the appropriate scenario. GPy backend.

    Args:
        scenario (BenchmarkFunctions): The test benchmark scenario
        X_init (np.ndarray): Array of initial test point of shape [num_init_points, input_dim]
        Y_init (np.ndarray): Array of objectives at the intitial test points of shape [num_init_points, 1]
        n_restarts: controls the hyperparameter optimization

    Returns:
        GPyModelWrapper: A model for the scenario
    """
    input_dim = X_init.shape[1]  # X_init of shape [num_init_points, input_dim]
    if scenario in {BenchmarkFunctions.SIMPLE_ADDITIVE, BenchmarkFunctions.HARD_ADDITIVE}:
        # Use an additive kernel ..
        raise NotImplementedError
    elif scenario in {BenchmarkFunctions.COSINES_ADDITIVE}:
        kernel_parts = [GPy.kern.RBF(1, active_dims=[i]) for i in range(input_dim)]
        kernel = Add(kernel_parts)
    elif scenario in {BenchmarkFunctions.COSINES_SYMMETRIC}:
        kernel = GPy.kern.RBF(input_dim)
        kernel = Symmetric(kernel, transform=np.array(-1.0))
    elif scenario in {BenchmarkFunctions.COSINES_ADDITIVE_SYMMETRIC}:
        kernel_parts = [GPy.kern.RBF(1, active_dims=[i]) for i in range(input_dim)]
        kernel = Add(kernel_parts)
        kernel = Symmetric(kernel, transform=np.array(-1.0))
    else:
        kernel = GPy.kern.RBF(input_dim)
    gpy_model = GPy.models.GPRegression(X_init, Y_init, kernel=kernel)
    gpy_model.Gaussian_noise.constrain_fixed(NOISE_LEVEL)
    model = GPyModelWrapper(gpy_model, n_restarts=n_restarts)
    return model


def _lengthscale_prior(lower: float, upper: float, dim: int) -> gpytorch.priors.Prior:
    """Lengthscale prior for a `dim`-dimensional cube with edge (lower, upper).

    Args:
        lower: lower bound of the parameter space
        upper: upper bound of the parameter space
        dim: dimension of the problem

    Returns:
        lengthscale prior
    """
    if upper <= lower:
        raise ValueError(f"Upper bound must be greater than lower bound: {lower} < {upper}.")

    # Use the prior over the interval (a, b). Both a and b are controlled by "characteristic length" of the problem
    characteristic_length = (upper - lower) * dim ** 0.5
    a = 1e-3 * characteristic_length
    b = 10 * characteristic_length

    sigma = a / 10  # There is only a *smoothed* box prior, so there is a smoothing hyperparameter
    return gpytorch.priors.SmoothedBoxPrior(a, b, sigma=sigma)


def _lengthscale_prior_from_parameter_space(parameter_space: ParameterSpace) -> gpytorch.priors.Prior:
    """A lengthscale prior for a box with different edges.

    Args:
        parameter_space: parameter space

    Returns:
        lengthscale prior

    See also:
        _lengthscale_prior, if all the edges are equal
    """
    lower = min(bounds[0] for bounds in parameter_space.get_bounds())
    upper = max(bounds[1] for bounds in parameter_space.get_bounds())
    return _lengthscale_prior(lower=lower, upper=upper, dim=parameter_space.dimensionality)


def _additive_kernel(parameter_space: ParameterSpace) -> gpytorch.kernels.Kernel:
    """Additive kernel, with a separate Matern kernel for each dimension"""
    kernels = []

    # We define a separate kernel for each dimension
    for i, (lower, upper) in enumerate(parameter_space.get_bounds()):
        prior = _lengthscale_prior(lower=lower, upper=upper, dim=1)
        k = gpytorch.kernels.MaternKernel(active_dims=torch.tensor([i]), lengthscale_prior=prior)
        kernels.append(k)

    kernel = gpytorch.kernels.AdditiveKernel(*kernels)
    return kernel


def _gpytorch_model_factory(
    x: torch.Tensor, y: torch.Tensor, scenario: BenchmarkFunctions, parameter_space: ParameterSpace
) -> mw.GPModel:
    """Constructs a GP model.

    Args:
        x: inputs, shape (n_samples, input_dim)
        y: outputs, shape (n_samples)
        scenario: which benchmark
        parameter_space: parameter space

    Returns:
        GP model

    Note:
        The default is the Matern52 kernel. For the COSINES_ADDITIVE scenario, we use an additive kernel.
    """
    if scenario in {BenchmarkFunctions.COSINES_ADDITIVE}:
        kernel = _additive_kernel(parameter_space)
    elif scenario in {BenchmarkFunctions.COSINES_ADDITIVE_SYMMETRIC}:
        base_additive_kernel = _additive_kernel(parameter_space)
        kernel = AxisReflectiveSymmetricKernel(base_kernel=base_additive_kernel)
    elif scenario in {BenchmarkFunctions.COSINES_SYMMETRIC}:
        base_kernel = gpytorch.kernels.MaternKernel(
            lengthscale_prior=_lengthscale_prior_from_parameter_space(parameter_space)
        )
        kernel = AxisReflectiveSymmetricKernel(base_kernel=base_kernel)
    else:
        kernel = gpytorch.kernels.MaternKernel(
            lengthscale_prior=_lengthscale_prior_from_parameter_space(parameter_space)
        )

    # Fix the noise to be small (we use deterministic functions)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(0.5 * NOISE_LEVEL, NOISE_LEVEL)
    )
    scaled_kernel = gpytorch.kernels.ScaleKernel(kernel)

    return mw.GPModel(train_x=x, train_y=y, likelihood=likelihood, covar_module=scaled_kernel)


def get_model_for_scenario_gpytorch(
    X_init: np.ndarray,
    Y_init: np.ndarray,
    scenario: BenchmarkFunctions,
    n_restarts: int = 5,
) -> mw.GPyTorchModelWrapper:
    """Get a model from initial data points for the appropriate scenario. GPyTorch backend.

    Args:
        X_init (np.ndarray): Array of initial test point of shape [num_init_points, input_dim]
        Y_init (np.ndarray): Array of objectives at the intitial test points of shape [num_init_points, 1]
        scenario (BenchmarkFunctions): The test benchmark scenario
        n_restarts: controls the hyperparameter optimization

    Returns:
        GPyTorchModelWrapper: A model for the scenario
    """
    x = mw._map_to_tensor(X_init)
    y = mw._outputs_remove_dim(Y_init)

    # Get the parameter space
    input_dim = X_init.shape[1]
    _, space = BenchmarkFunctions.get_function_and_space(function_name=scenario.value, dimensionality=input_dim)
    if space.dimensionality != input_dim:
        raise ValueError(f"Dimensions don't match: {space.dimensionality} != {input_dim}.")

    model = _gpytorch_model_factory(x=x, y=y, scenario=scenario, parameter_space=space)
    return mw.GPyTorchModelWrapper(model, n_restarts=n_restarts)
