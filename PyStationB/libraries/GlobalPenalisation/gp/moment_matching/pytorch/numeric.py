# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""PyTorch numerical utilitiers"""
from typing import Tuple
import numpy as np
import torch
from torch.cuda.amp import custom_bwd, custom_fwd


class PassthroughGradientClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, x, minimum, maximum):
        return x.clamp(min=minimum, max=maximum)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def passthrough_clamp(x: torch.Tensor, minimum, maximum) -> torch.Tensor:
    return PassthroughGradientClamp.apply(x, minimum, maximum)


def normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """The PDF of the standard normal (zero-mean, unit-variance)"""
    return torch.exp(-(x ** 2 / 2)) / np.sqrt(2 * np.pi)


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """The CDF of the standard normal distribution (zero-mean, unit-variance)"""
    return torch.distributions.Normal(0, 1.0).cdf(x)


def correlation_and_std_from_covariance(covariance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns covariance normalised into correlation, and the corresponding standard deviations.

    Args:
        covariance: covariance matrix, shape (n, n)

    Returns:
        correlation: correlation matrix, shape (n, n)

    Note:
        when diagonal entries of the covariance matrix vanish, this implementation of correlation
        becomes numerically unstable
    """
    std = torch.sqrt(torch.diagonal(covariance, dim1=-2, dim2=-1))  # A vector of standard deviations
    outer_std = std[..., None] * std[..., None, :]  # A matrix A of the form A_xy = std_x * std_y
    correlation = covariance / outer_std

    correlation = passthrough_clamp(correlation, -1.0, 1.0)

    return correlation, std


def correlation_between_distinct_sets_from_covariance(
    covariance: torch.Tensor, std1: torch.Tensor, std2: torch.Tensor
) -> torch.Tensor:
    """Returns the correlation matrix between two sets of random variables from the covariance matrix between them,
    and the standard deviations of the variables in each set.

    Args:
        covariance: covariance matrix, shape (..., n1, n2)
        std1: standard deviations of the first set of random variables of shape (..., n1)
        std2: standard deviations of the second set of random variables of shape (..., n2)

    Returns:
        correlation: correlation matrix, shape (..., n1, n2) (same shape as covariance)
    """
    outer_std = std1[..., None] * std2[..., None, :]
    correlation = covariance / outer_std

    correlation = passthrough_clamp(correlation, -1.0, 1.0)

    return correlation


def expected_improvement(y_min: torch.Tensor, mean: torch.Tensor, standard_deviation: torch.Tensor) -> torch.Tensor:
    """Expected improvement.

    .. math::

        EI = \\mathbb E(max(y_min - f(x), 0))

    Args:
        y_min: observed global minimum
        mean: mean of the random variable
        standard_deviation: standard deviation of the random variable

    Returns:
        Expected improvement
    """
    y_normalised = (y_min - mean) / standard_deviation

    return standard_deviation * (y_normalised * normal_cdf(y_normalised) + normal_pdf(y_normalised))


def lower_confidence_bound(beta: torch.Tensor, mean: torch.Tensor, standard_deviation: torch.Tensor) -> torch.Tensor:
    """The *Lower* Confidence Bound acquisition function for a Gaussian variable.

    TODO: implementation same as jax, move into shared numeric?

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
