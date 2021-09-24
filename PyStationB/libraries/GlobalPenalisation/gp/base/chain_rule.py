# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import numpy as np


def chain_rule_means_vars(d_acq_dy: np.ndarray, dy_dx: np.ndarray) -> np.ndarray:
    """Implements the chain rule with respect to the means/standard deviation vectors for candidate points.

    Args:
        d_acq_dy: gradient of acquisition with respect to one-dimensional variable (e.g. mean, var or std).
            Shape (n_candidates, 1)
        dy_dx: gradient of the variable with respect to the inputs. Shape (n_candidates, n_inputs)

    Returns:
        d_acq_dx, shape (n_candidates, n_inputs). Note that it's not the whole expression, if ``acq`` depends on other
            variable than ``y`` as well
    """
    return d_acq_dy * dy_dx


def chain_rule_cross_covariance(d_acq_d_cov: np.ndarray, d_cov_dx: np.ndarray) -> np.ndarray:
    """Implements the chain rule with respect to the cross-covariance matrix between candidate points and selected points.

    Args:
        d_acq_d_cov: gradient of acquisition with respect to covariance matrix between candidates and selected points.
            Shape (n_candidates, n_selected)
        d_cov_dx: gradient of covariance matrix between candidates and selected points with respect to the inputs.
            Shape (n_candidates, n_selected, n_inputs)

    Returns:
        d_acq_dx, shape (n_candidates, n_selected).
            Note that it's not the whole expression, if ``acq`` depends on other variable than ``cov`` as well
    """
    return np.einsum("ij,ijk -> ik", d_acq_d_cov, d_cov_dx)


def chain_rule_means_from_predict_joint(d_acq_d_means: np.ndarray, d_means_dx: np.ndarray) -> np.ndarray:
    """
    Args:
        d_acq_d_means: gradient of acquisition with respect to the means vector. Shape (n_points, 1).
        d_means_dx: gradient of the means vector with respect to the inputs vector.
            Shape (n_points, n_points, input_dim)

    Returns:
        part of d_acq_dx, which can be calculated from the chain rule with respect to the means vector.
        Shape (n_points, input_dim)
    """
    d_acq_d_means = d_acq_d_means.ravel()
    return np.einsum("i,ikl", d_acq_d_means, d_means_dx)


def chain_rule_covariance(d_acq_d_covariance: np.ndarray, d_covariance_dx: np.ndarray) -> np.ndarray:
    """Chain rule for the gradient with respect to covariance.

    Args:
        d_acq_d_covariance: gradients of the acquisition function with respect to the covariance.
            Shape (n_points, n_points)
        d_covariance_dx: gradients of the covariance matrix entries with respect to the inputs.
            Shape (n_points, n_points, n_points, input_dim)

    Returns:
        part of d_acq_dx, which can be calculated from the chain rule with respect to covariance.
        Shape (n_points, input_dim)
    """
    return np.einsum("ij,ijkl", d_acq_d_covariance, d_covariance_dx)
