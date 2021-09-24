# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from typing import Tuple

import GPy
import numpy as np
from emukit.model_wrappers import GPyModelWrapper

import gp.numeric as numeric


def get_gpy_model(input_dim: int, n_points: int = 3) -> GPyModelWrapper:
    """Returns a dummy Gaussian Process on `n_training_points`, where the input space has `input_dim` dimensions."""
    X = np.random.rand(n_points, input_dim)
    Y = sum(np.sin(X[:, i]) for i in range(input_dim))
    Y = Y.reshape((-1, 1))
    gpy_model = GPy.models.GPRegression(X, Y)
    emukit_model = GPyModelWrapper(gpy_model)
    return emukit_model


def evaluate_model(model: GPyModelWrapper, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates `model` at points `x`.

    Args:
        model: Gaussian Process
        x: points to evaluate the model at, shape (n_training_points, input_dim)

    Returns:
        mean vector, shape (n_training_points,)
        standard deviation vector, shape (n_training_points)
        correlations, shape (n_training_points, n_training_points)
    """
    means, covariance = model.predict_with_full_covariance(x)
    means = means.ravel()
    std_dev = numeric.covariance_into_standard_deviations(covariance)
    correlation = numeric.correlation_from_covariance(covariance)
    return means, std_dev, correlation
