# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This module implements global penalization using the moment-matching strategy introduced in

C.E. Clark, The Greatest of a Finite Set of Random Variables, Operations Research, Vol. 9, No. 2 (1961).
"""
from typing import List, Tuple

import numpy as np
import torch
from gp.moment_matching.common import calc_cum_min_mean, calc_cum_min_var
from gp.moment_matching.pytorch.numeric import normal_cdf, normal_pdf, passthrough_clamp


def calc_beta_sqrt(var1: torch.Tensor, var2: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """Helper function to calculate :math:`\beta` in the equations for moment approximation to :math:`\\max_i Y_i`.

    Args:
        var1: Variance of Gaussian-distributed random variables Y_i
        var2: Variance of theta_j=min_{i in 1:j} Y_i
        cov: The vector of covariances between Gaussian random variable Y_i and
            theta_j=min_j Y_j

    Returns:
        Square-root of beta in equations for moment-matching (TODO need ref.).
    """
    beta = var1 + var2 - 2 * cov
    # Ensure that beta >= 0. (sometimes it can become negative due to numerical errors)
    beta_clipped = passthrough_clamp(beta, 0.0, None)
    return torch.sqrt(beta_clipped)


def calc_alpha_pdf_cdf(
    mean1: torch.Tensor, mean2: torch.Tensor, beta_sqrt: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate Gaussian PDF and CDF function of `\alpha` in the equations for moment-matching approximation to
    :math:`\\max_i Y_i`.

    Args:
        mean1: Mean of Gaussian random variable Y_i
        mean2: Mean of another Gaussian random variable Y_j
        beta_sqrt (torch.Tensor): Tensor of same shape as mean1 and mean2 – the square root of beta
            for Y_i and Y_j calculated in calc_beta_sqrt()

    Returns:
        Two tensors of same shape as mean1, mean2 and beta_sqrt representing Gaussian PDF(alpha) and CDF(alpha) where
            alpha is described in (TODO need math ref.)
    """
    alpha_nominator = mean1 - mean2
    # Account for the case when beta_i == 0
    alpha_limiting_case = torch.where(
        alpha_nominator > 0.0, torch.tensor(np.inf, dtype=torch.float64), torch.tensor(-np.inf, dtype=torch.float64)
    )
    alpha = torch.where(beta_sqrt > 0.0, alpha_nominator / beta_sqrt, alpha_limiting_case)

    alpha_pdf = normal_pdf(alpha)
    alpha_cdf = normal_cdf(alpha)
    return alpha_pdf, alpha_cdf


def calc_output_to_cum_min_cov(
    cross_covariance: torch.Tensor,
    prob_is_min: torch.Tensor,
) -> torch.Tensor:
    """
    Given two sets of jointly Gaussian random variables [Y_1, ..., Y_d] and [Z_1, ..., Z_m],
    calculate the approximate covariance between min(Y_1, ..., Y_d) and [Z_1, ..., Z_m].

    Args:
        cross_covariance: Covariance between Gaussian variables [Y_1, ..., Y_d] and [Z_1, ..., Z_m]
            of shape [batch_dim..., d, m]
        prob_is_min: (approximate) probability that each of [Y_1, ..., Y_d] is the
            minimum. Shape [batch_dim..., d]

    Returns:
        torch.Tensor: Approximate covariance between min(Y_1, ..., Y_d) and [Z_1, ..., Z_m].
            Shape [batch_dim..., m]
    """
    return (cross_covariance * prob_is_min[..., None]).sum(dim=-2)


def get_next_cumulative_min_moments(
    mean: torch.Tensor,
    variance: torch.Tensor,
    cov_to_next: torch.Tensor,
    cum_min_mean: torch.Tensor,
    cum_min_var: torch.Tensor,
    prob_is_min: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For the next output Z, given a Gaussian approximation to min(Y_1, ..., Y_j), calculate the
    moments of cumulative minimum  min(Y_1, ..., Y_j, Z)

    Can operate on a batch of new suggested outputs.

    Args:
        mean: Tensor of shape [batch_dim...] the mean of the next Gaussian Z
        variance: Tensor of shape [batch_dim...] with the variances of the next Gaussian Z
        cov_to_next: Array of shape [batch_dim..., j] where each element is the covariance of the
            next output Z to one of the j previous ones Y_i (for i = 1 ... j).
        cum_min_mean: (approximate) mean of cumulative minimum min(Y_1, ..., Y_j)
        cum_min_var: (approximate) variance of cumulative minimum min(Y_1, ..., Y_j)
        # alpha_pdfs: Sequence of arrays of shape [N] or [1] with previous PDF(alpha) parameters
        # alpha_cdfs: Sequence of arrays of shape [N] or [1] with previous CDF(alpha) parameters
        prob_is_min: Approximate probability (product of CDF(alpha)) that each of [Y_1, ..., Y_j]
            is the minimum.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Returns four arrays
            - next_cum_min_mean: the mean of the cumulative minimum up to index next_output_idx of shape [num_points]
            - next_cum_min_var: the variance of the cumulative min. up to index next_output_idx of shape [num_points]
            - new_output_cum_min_cov: Array of vectors representing the covariance between the next output
                and previous cumulative minima. Has shape [next_output_idx], and each element is a vector of
                covariances of shape [num_points]
            - alpha_pdf: Array of shape [num_points] of the standard normal PDF of helper values alpha, which are
                useful for later calculations of the moments.
            - alpha_cdf: Array of shape [num_points] of the standard normal CDF of helper values alpha, which are
                useful for later calculations of the moments.
    """
    # Assert the shapes are consistent
    assert mean.ndim == variance.ndim
    assert len({mean.shape, variance.shape, cov_to_next.shape[: mean.ndim]}) == 1, "Batch dimensions must be equal"

    # if isinstance(cum_min_mean, float) or cum_min_mean.shape[0] == 1:
    #     # If only a single set of previous points given, and the cumulative minimum moments are to be calculated
    #     # for a batch of new candidate outputs:
    #     assert isinstance(cum_min_var, float) or cum_min_var.shape[0] == 1

    new_output_cum_min_cov = calc_output_to_cum_min_cov(
        cross_covariance=cov_to_next[..., None], prob_is_min=prob_is_min
    ).squeeze(-1)

    # Calculate alphas and betas
    beta_sqrt = calc_beta_sqrt(
        var1=cum_min_var,
        var2=variance,
        cov=new_output_cum_min_cov,
    )
    alpha_pdf, alpha_cdf = calc_alpha_pdf_cdf(mean1=cum_min_mean, mean2=mean, beta_sqrt=beta_sqrt)
    # Calculate the moments
    next_cum_min_mean = calc_cum_min_mean(
        mean=mean,
        cum_min_prev_mean=cum_min_mean,
        alpha_pdf=alpha_pdf,
        alpha_cdf=alpha_cdf,
        beta_sqrt=beta_sqrt,
    )
    next_cum_min_var = calc_cum_min_var(
        mean=mean,
        variance=variance,
        cum_min_mean=next_cum_min_mean,
        cum_min_prev_mean=cum_min_mean,
        cum_min_prev_var=cum_min_var,
        alpha_pdf=alpha_pdf,
        alpha_cdf=alpha_cdf,
        beta_sqrt=beta_sqrt,
    )
    return next_cum_min_mean, next_cum_min_var, new_output_cum_min_cov, alpha_pdf, alpha_cdf


def calculate_cumulative_min_moments(
    means: torch.Tensor,  # [..., N]
    covariance: torch.Tensor,  # [..., N, N]
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Calculate approximate moments of the cumulative minimum of multiple Gaussians:
        cum_min_j = min_{i <= j} Y_i
    where cum_min_j is the maximum of j Gaussian variables Y_i for i = 1, ..., j.

    Args:
        means (torch.Tensor): Array of shape [batch_size, N] representing the means of the Gaussian variables.
        corr_matrix (torch.Tensor): Array of shape [batch_size, N, N] representing the covariance between the
            N Gaussian variables in each batch

    Returns:
        - List of tensors with the (approximations to) means of cumulative minima
        - List of tensors with the (approximations to) variances of cumulative minima
        - List of tensors with the intermediate PDF(alpha) values
        - List of tensors with the intermediate CDF(alpha) values
    """
    # Assert the shapes are consistent
    assert len({means.shape[-1], covariance.shape[-2], covariance.shape[-1]}) == 1

    # Define the arrays to iteratively compute (these are used as indexable variables)
    cum_min_means: List[torch.Tensor] = []
    cum_min_vars: List[torch.Tensor] = []

    prob_is_mins: List[torch.Tensor] = [torch.reshape(torch.tensor(1.0, dtype=means.dtype), [1] * means.ndim)]

    # First entries are exact
    cum_min_means.append(means[..., 0])
    cum_min_vars.append(covariance[..., 0, 0])
    for i in range(1, means.shape[-1]):
        next_cum_min_mean, next_cum_min_var, _, next_alpha_pdf, next_alpha_cdf = get_next_cumulative_min_moments(
            mean=means[..., i],
            variance=covariance[..., i, i],
            cov_to_next=covariance[..., :i, i],
            cum_min_mean=cum_min_means[-1],
            cum_min_var=cum_min_vars[-1],
            prob_is_min=prob_is_mins[-1],
        )
        #  Calculate (approximate) prob. that each of Y_0, ..., Y_i is the minimum
        next_prob_is_min = torch.cat(
            (prob_is_mins[-1] * (1.0 - next_alpha_cdf[..., None]), next_alpha_cdf[..., None]), dim=-1
        )

        cum_min_means.append(next_cum_min_mean)
        cum_min_vars.append(next_cum_min_var)
        prob_is_mins.append(next_prob_is_min)
    return cum_min_means, cum_min_vars, prob_is_mins


def approximate_minimum_with_prob_is_min(
    means: torch.Tensor,  #  [..., N]
    covariance: torch.Tensor,  # [..., N, N]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the approximate mean and variance of the minimum of N Gaussian variables
    with the given means and covariance. Also return the (approximate) prob. that each variable
    is the minimum (useful for downstream approximations involving the minimum).

    Note: The 0th dimension is the batch dimension.

    Args:
        means (torch.Tensor): Array of shape [batch_size, N] representing the means of the Gaussian variables.
        covariance (torch.Tensor): Array of shape [batch_size, N, N] representing the covariance between the
            N Gaussian variables in each batch

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two arrays of shape [batch_size] – the mean and
        the variance of the minimum of the N Gaussian variables in each batch row.
    """
    cum_min_means, cum_min_vars, prob_is_mins = calculate_cumulative_min_moments(
        means=means,
        covariance=covariance,
    )
    return cum_min_means[-1], cum_min_vars[-1], prob_is_mins[-1]
