# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from typing import Tuple

import torch

from gp.moment_matching.pytorch.numeric import passthrough_clamp, normal_pdf, normal_cdf


def approximate_min_of_first_2_elems(
    means: torch.Tensor,
    covariance: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given means and covariance of a d-dimensional multivariate normal vector [Y_1, ... Y_d],
    return mean and covariance matrix of a Gaussian approximation to:
        [min(Y_1, Y_2), Y_3, ..., Y_d]
    """
    assert means.shape[-1] >= 2, "Need at least 2 element to take minimum over"
    #  cdf(alpha) is the prob. of Y_2 being the minimum of (Y_1, Y_2)
    pdf_alpha, cdf_alpha, sigma = _calc_alpha_pdf_cdf_and_sigma(means[..., :2], covariance[..., :2])

    min_mean = mean_of_min_of_2(
        means=means[..., :2],
        covariance=covariance[..., :2, :2],
        pdf_alpha=pdf_alpha,
        cdf_alpha=cdf_alpha,
        sigma=sigma,
    )
    min_2nd_moment = second_moment_of_min_of_2(
        means=means[..., :2], covariance=covariance[..., :2, :2], pdf_alpha=pdf_alpha, cdf_alpha=cdf_alpha, sigma=sigma
    )
    min_variance = min_2nd_moment - min_mean ** 2
    cross_covariance = cross_covariance_to_min(
        cross_covariance=covariance[..., :2, 2:],
        cdf_alpha=cdf_alpha,
    )
    #  Update means and covariance
    means = torch.cat((min_mean[..., None], means[..., 2:]), dim=-1)
    covariance = stack_covariance_from_subcovariances(
        cov1=min_variance[..., None, None], cov2=covariance[..., 2:, 2:], cross_cov=cross_covariance[..., None, :]
    )
    return means, covariance


def approximate_min_of_first_n_elems(
    means: torch.Tensor,
    covariance: torch.Tensor,
    n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given means and covariance of a d-dimensional multivariate normal vector [Y_1, ... Y_d],
    return mean and covariance matrix of a Gaussian approximation to:
        [min(Y_1, ..., Y_n), Y_{n+1}, ..., Y_d]
    """
    assert n <= means.shape[-1], "Number of elem. to take minimum over cannot be more than total num. of elem."

    for i in range(n - 1):
        #  Approximate [min(Z_1, Z_2), Z_3, ...] from the previous approx [Z_1, Z_2, Z_3, ...]
        means, covariance = approximate_min_of_first_2_elems(means, covariance)
    return means, covariance


def approximate_minimum(means: torch.Tensor, covariance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, cov = approximate_min_of_first_n_elems(means=means, covariance=covariance, n=means.shape[-1])
    #  Reshape the (1D) covariance matrix into variance
    var = cov.squeeze(-1)
    return mean, var


def mean_of_min_of_2(
    means: torch.Tensor,
    covariance: torch.Tensor,
    pdf_alpha: torch.Tensor,
    cdf_alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    return means[..., 0] * (1.0 - cdf_alpha) + means[..., 1] * (cdf_alpha) - sigma * pdf_alpha


def second_moment_of_min_of_2(
    means: torch.Tensor,
    covariance: torch.Tensor,
    pdf_alpha: torch.Tensor,
    cdf_alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    return (
        (means[..., 0] ** 2 + covariance[..., 0, 0]) * (1.0 - cdf_alpha)
        + (means[..., 1] ** 2 + covariance[..., 1, 1]) * cdf_alpha
        - (means[..., 0] + means[..., 1]) * sigma * pdf_alpha
    )


def cross_covariance_to_min(
    cross_covariance: torch.Tensor,
    cdf_alpha: torch.Tensor,
) -> torch.Tensor:
    """Given cross-covariance between two (jointly Gaussian) random vectors [Y_1, Y_2]
    and [X_1, ..., X_D], calculate the cross-covariance between [min(Y_1, Y_2)] and [X_1, ..., X_D]

    Args:
        cross_covariance: Cross-covariance between 2 Gaussians [Y_1, Y_2] and D
            other Gaussians [X_1, ..., X_D]. Shape [batch_dim..., 2, D]
        cdf_alpha: The CDF(alpha) helper value of shape [batch_dim...] calculated when computing mean/variance
        of min(Y_1, Y_2)

    Returns:
        Cross-covariance of shape [batch_dim..., D] between min(Y_1, Y_2) and [X_1, ..., X_D]
    """

    return (
        cross_covariance[..., 0, :] * (1.0 - cdf_alpha[..., None]) + cross_covariance[..., 1, :] * cdf_alpha[..., None]
    )


def _calc_alpha_pdf_cdf_and_sigma(means: torch.Tensor, covariance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha, sigma = _calc_alpha_sigma(means, covariance)

    pdf_alpha = normal_pdf(alpha)
    cdf_alpha = normal_cdf(alpha)
    return pdf_alpha, cdf_alpha, sigma


def _calc_alpha_sigma(means: torch.Tensor, covariance: torch.Tensor, eps=1e-10) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha_nominator = means[..., 0] - means[..., 1]
    sigma_sq = covariance[..., 0, 0] + covariance[..., 1, 1] - 2 * covariance[..., 0, 1]
    #  Ensure denominator strictly positive (can be negative due to numerical inacuracies)
    sigma_sq = passthrough_clamp(sigma_sq, eps, None)
    sigma = torch.sqrt(sigma_sq)

    alpha = alpha_nominator / sigma
    return alpha, sigma


def stack_covariance_from_subcovariances(
    cov1: torch.Tensor, cov2: torch.Tensor, cross_cov: torch.Tensor
) -> torch.Tensor:
    """Given covariances of two sets of points, and the cross-covariance between them, reconstruct
    the full covariance matrix

    Args:
        cov1 (torch.Tensor): Covariance of 1st set of points. Shape [..., N1, N1]
        cov2 (torch.Tensor): Covariance of 2nd set of points. Shape [..., N2, N2]
        cross_cov (torch.Tensor): Cross-covariance of shape [..., N1, N2]

    Returns:
        torch.Tensor: [description]
    """
    top = torch.cat((cov1, cross_cov), dim=-1)
    bottom = torch.cat((cross_cov.transpose(-2, -1), cov2), dim=-1)
    return torch.cat((top, bottom), dim=-2)
