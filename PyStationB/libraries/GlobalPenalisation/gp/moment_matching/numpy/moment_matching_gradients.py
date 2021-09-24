# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This module implements the gradients for the moment-matching approximation to the minimum introduced in:

C.E. Clark, The Greatest of a Finite Set of Random Variables, Operations Research, Vol. 9, No. 2 (1961).

using numpy.

TODO: Incomplete - gradients not implemented in raw numpy yet
"""
import numpy as np
from scipy.stats import norm

# from typing import Tuple


def beta_sqrt_grad(
    std: np.ndarray, theta_std_prev: np.ndarray, output_to_prev_theta_corr: np.ndarray, beta_sqrt: np.ndarray
) -> np.ndarray:
    """Helper function to calculate :math:`\beta` in the equations for moment approximation to :math:`\\max_i Y_i`.

    Args:
        std (np.ndarray): Standard deviations of Gaussian-distributed random variables Y_i
        theta_std_prev (np.ndarray): The standard deviations of theta_j=max_j Y_j computed so far.
        output_to_prev_theta_corr (np.ndarray): The vector of correlations between Gaussian random variable Y_i and
            theta_j=max_j Y_j for

    Returns:
        np.ndarray: The beta in equations for moment-matching.
    """
    d_theta_std_prev = 2 * theta_std_prev - 2 * std * output_to_prev_theta_corr
    d_std = 2 * std - 2 * theta_std_prev * output_to_prev_theta_corr
    d_output_to_prev = -2 * theta_std_prev * std
    return d_std, d_theta_std_prev, d_output_to_prev


def alpha_pdf_cdf_grad(
    mean: np.ndarray, theta_mean_prev: np.ndarray, beta: np.ndarray, alpha_pdf: np.ndarray, alpha_cdf: np.ndarray
) -> np.ndarray:
    # -- First for the cdf:
    # Stable gradient for beta: if beta == 0 the gradient is 0.
    d_beta = np.where(beta > 0, -0.5 * (theta_mean_prev - mean) * alpha_pdf / (beta ** 1.5), 0)
    with np.errstate(divide="ignore"):
        beta_to_minus_half = 1 / np.sqrt(beta)  # Reuse computation of 1 / sqrt(beta)
    d_theta_mean_prev = np.where(beta > 0, beta_to_minus_half * alpha_pdf, 0)
    d_mean = np.where(beta > 0, -beta_to_minus_half * alpha_pdf, 0)
    # -- Add gradient from alpha_pdf:
    d_beta += np.where(beta > 0, (theta_mean_prev - mean) * beta_to_minus_half * d_beta, 0)
    d_theta_mean_prev += np.where(beta > 0, (theta_mean_prev - mean) * beta_to_minus_half * d_theta_mean_prev, 0)
    d_mean += np.where(beta > 0, (theta_mean_prev - mean) * beta_to_minus_half * d_mean, 0)
    return d_mean, d_theta_mean_prev, d_beta


def output_i_theta_j_corr_grad(
    std, corr, theta_std, theta_prev_std, output_theta_prev_corr, alpha_pdf, alpha_cdf, output_i_theta_j_corr
):
    """
    Computes the (approximate) correlation between output Y_i and theta_j (the cumulative minimum of outputs
    up to index j), where i > j.

    Args:
        std: Standard deviation of the Gaussian variable Y_j
        corr: Correlation between Gaussian variables Y_i and Y_j
        theta_std: Standard deviation of the cumulative minimum up to index j
        theta_prev_std: Standard deviation of the cumulative minimum up to index j - 1
        output_theta_prev_corr ([type]): Correlation between Gaussian variable Y_i and cumulative min. up
            to index j - 1
        alpha: Helper alpha variable for output with index j (see calc_alpha)

    Returns:
        np.ndarray: Approximate correlation between Gaussian variable Y_i and theta_j (the cumulative minimum up to
            index j)
    """
    d_std = corr * alpha_cdf / theta_std
    d_corr = std * alpha_cdf / theta_std

    d_theta_std = -1 * output_i_theta_j_corr / theta_std

    d_theta_prev_std = output_theta_prev_corr * (1 - alpha_cdf) / theta_std
    d_output_theta_prev_corr = theta_prev_std * (1 - alpha_cdf) / theta_std

    d_alpha_cdf = (-theta_prev_std * output_theta_prev_corr + std * corr) / theta_std

    return d_std, d_corr, d_theta_std, d_theta_prev_std, d_output_theta_prev_corr, d_alpha_cdf


def theta_mean_grad(mean, theta_mean_prev, alpha_pdf, alpha_cdf, beta, theta_mean) -> np.ndarray:
    d_mean = alpha_cdf
    d_theta_mean_prev = 1 - alpha_cdf  # Same as norm.cdf(-alpha)

    d_alpha_pdf = -np.sqrt(beta)
    d_alpha_cdf = mean - theta_mean_prev

    d_beta = 0.5 * alpha_pdf / np.sqrt(beta)
    return d_mean, d_theta_mean_prev, d_alpha_pdf, d_alpha_cdf, d_beta


def calc_theta_uncentered_2nd_moment(mean, std, theta_mean_prev, theta_std_prev, alpha, beta):
    term1 = (theta_mean_prev ** 2 + theta_std_prev ** 2) * norm.cdf(-alpha)
    term2 = (mean ** 2 + std ** 2) * norm.cdf(alpha)
    term3 = -(theta_mean_prev + mean) * np.sqrt(beta) * norm.pdf(alpha)
    theta_uncentered_2nd_moment = term1 + term2 + term3
    return theta_uncentered_2nd_moment


# def theta_std_grad(mean, std, theta_mean, theta_mean_prev, theta_std_prev, alpha_pdf, alpha_cdf, beta, theta_std):
#     d_mean = 2 * mean * alpha_cdf - np.sqrt(beta) * alpha_pdf
#     d_std = 2 * std * alpha_cdf
#     d_theta_mean_prev = 2 * theta_mean_prev * (1 - alpha_cdf) - beta_sqrt * alpha_pdf
#     d_theta_std_prev = 2 * theta_std * (1 - alpha_cdf)

#     d_beta = 0
#     theta_uncentered_2nd_moment = calc_theta_uncentered_2nd_moment(
#         mean=mean, std=std, theta_mean_prev=theta_mean_prev, theta_std_prev=theta_std_prev, alpha=alpha, beta=beta
#     )
#     theta_var = theta_uncentered_2nd_moment - theta_mean ** 2
#     return np.sqrt(theta_var)


# def get_next_cumulative_min_moments(
#     next_output_idx: int,
#     mean: np.ndarray,
#     std: np.ndarray,
#     prev_stds: np.ndarray,
#     corr_to_next: np.ndarray,
#     theta_means: np.ndarray,
#     theta_stds: np.ndarray,
#     alphas: np.ndarray,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     For the next output Y_i, calculate the moments of cumulative minimum up to index i:
#         theta_i = min_{j <= i} Y_j
#     from the moments of cumulative minimum up to index i - 1.

#     Can operate on a batch of new suggested outputs at index next_output_idx.

#     Args:
#         next_output_idx (int): Integer index of the next output to incorporate into the cumulative minimum. Equals
#             the number of previous points (num_prev_points)
#         mean (np.ndarray): Array of shape [N] with the means of the next output(s) Y_i
#         std (np.ndarray): Array of shape [N] with the standard deviations of the next output(s) Y_i
#         prev_stds (np.ndarray): Array of shape [N, num_prev_points] or [1, next_output_idx].
#             If first dimension has size 1, then the batch of next outputs of size [N] is treated as N candidate point
#             suggestions for the next point, and the cumulative minimum will be calculated with respect to the same
#             single set of previous outputs.
#             Otherwise, if first dimension has size N, the batch of next outputs are treated as independent outputs,
#             and the cumulative minimum will be calculated with respect to a different set of previous outputs for each
#             of the next outputs
#         corr_to_next (np.ndarray): Array of shape [N, num_prev_points] where each element is the correlation of the
#             next output Y_i to one of the previous ones.
#         theta_means (np.ndarray): Sequence of arrays of shape [N] or [1] with previous cumulative minimum means
#         theta_stds (np.ndarray): Sequence of arrays of shape [N] or [1] with previous cumulative standard deviations
#         alphas (np.ndarray): Sequence of arrays of shape [N] or [1] with previous alpha parameters

#     Returns:
#         Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Returns four arrays
#             - next_theta_mean: the mean of the cumulative minimum up to index next_output_idx of shape [num_points]
#             - next_theta_std: the st. deviation of the cumulative min. up to index next_output_idx of
#                   shape [num_points]
#             - new_output_theta_corr: Array of vectors representing the correlation between the next output
#                 and previous cumulative minima. Has shape [next_output_idx], and each element is a vector of
#                 correlations of shape [num_points]
#             - alpha: Array of shape [num_points] of helper values alpha which are useful for later calculations
#                 of the moments.
#     """
#     assert mean.ndim == 1
#     assert std.ndim == 1
#     assert prev_stds.ndim == 2
#     assert corr_to_next.ndim == 2
#     # Assert the shapes are consistent
#     assert len({mean.shape[0], std.shape[0], corr_to_next.shape[0]}) == 1

#     if prev_stds.shape[0] == 1:
#         # If only a single set of previous points given, and the cumulative minimum moments are to be calculated
#         # for a batch of new candidate outputs:
#         assert isinstance(theta_means[next_output_idx - 1], float) or theta_means[next_output_idx - 1].shape[0] == 1
#         assert isinstance(theta_stds[next_output_idx - 1], float) or theta_stds[next_output_idx - 1].shape[0] == 1
#     else:
#         assert len({mean.shape[0], prev_stds.shape[0]}) == 1

#     # Array for correlations between the next output and previous theta_j (cumulative min. up to index j)
#     new_output_theta_corr = np.empty([next_output_idx], dtype=object)
#     # First correlation entry is exact
#     new_output_theta_corr[0] = corr_to_next[:, 0]
#     for j in range(1, next_output_idx):  # Iterate over previous thetas
#         new_output_theta_corr[j] = calc_output_i_theta_j_corr(
#             std=prev_stds[:, j],
#             corr=corr_to_next[:, j],
#             theta_std=theta_stds[j],
#             theta_prev_std=theta_stds[j - 1],
#             output_theta_prev_corr=new_output_theta_corr[j - 1],
#             alpha=alphas[j],
#         )
#     # Calculate alphas and betas
#     beta = calc_beta(
#         std=std,
#         theta_std_prev=theta_stds[next_output_idx - 1],
#         output_to_prev_theta_corr=new_output_theta_corr[next_output_idx - 1],
#     )
#     alpha = calc_alpha(mean=mean, theta_mean_prev=theta_means[next_output_idx - 1], beta=beta)
#     # Calculate the moments
#     next_theta_mean = calc_theta_mean(
#         mean=mean, theta_mean_prev=theta_means[next_output_idx - 1], alpha=alpha, beta=beta
#     )
#     next_theta_std = calc_theta_std(
#         mean=mean,
#         std=std,
#         theta_mean=next_theta_mean,
#         theta_mean_prev=theta_means[next_output_idx - 1],
#         theta_std_prev=theta_stds[next_output_idx - 1],
#         alpha=alpha,
#         beta=beta,
#     )
#     return next_theta_mean, next_theta_std, new_output_theta_corr, alpha


# def calculate_cumulative_min_moments(means, stds, corr_matrix):
#     """
#     Calculate approximate moments of the cumulative minimum of multiple Gaussians:
#         theta_j = min_{i <= j} Y_i
#     where theta_j is the maximum of j Gaussian variables Y_i for i = 1, ..., j.
#     """
#     assert means.ndim == 2
#     assert stds.ndim == 2
#     assert corr_matrix.ndim == 3
#     # Assert the shapes are consistent
#     assert len({means.shape[1], stds.shape[1], corr_matrix.shape[1], corr_matrix.shape[2]}) == 1
#     assert len({means.shape[0], stds.shape[0], corr_matrix.shape[0]}) == 1

#     ndims = means.shape[1]
#     # Define the arrays to iteratively compute (these are used as indexable variables)
#     theta_means = np.empty([ndims], dtype=object)
#     theta_stds = np.empty([ndims], dtype=object)
#     alphas = np.empty([ndims], dtype=object)
#     # First entries are exact
#     theta_means[0] = means[:, 0]
#     theta_stds[0] = stds[:, 0]
#     for i in range(1, means.shape[1]):
#         next_theta_mean, next_theta_std, next_output_theta_corr, next_alpha = get_next_cumulative_min_moments(
#             next_output_idx=i,
#             mean=means[:, i],
#             std=stds[:, i],
#             prev_stds=stds,
#             corr_to_next=corr_matrix[:, :, i],
#             theta_means=theta_means,
#             theta_stds=theta_stds,
#             alphas=alphas,
#         )
#         theta_means[i] = next_theta_mean
#         theta_stds[i] = next_theta_std
#         alphas[i] = next_alpha
#     return theta_means, theta_stds, alphas


# def correlation_from_covariance(covariance: np.ndarray, standard_deviations: Optional[np.ndarray]) -> np.ndarray:
#     """Compute the correlation matrices from an array of covariance matrices.

#     Args:
#         covariance (np.ndarray): Array of shape [num_cases, num_dim, num_dim] where (for every i)
#         the covariance[i, :, :] matrix is a positive semi-definite covariance matrix for a multivariate Gaussian.
#         standard_deviations: A [num_cases, num_dim] shaped array of standard deviations corresponding to the
#             covariance matrices. This argument is optional if standard deviations have been precomputed before calling

#     Returns:
#         np.ndarray: A [num_cases, num_dim, num_dim] shaped array of correlation matrices.
#     """
#     # stds_outer_product = standard_deviations[:, :, None] * standard_deviations[:, None, :]
#     stds_outer_product = np.einsum("...i,...j->...ij", standard_deviations, standard_deviations)
#     correlation = covariance / stds_outer_product
#     correlation[covariance == 0] = 0
#     # Due to numerical errors, sometimes abs(correlation) > 1.0 by a narrow margin
#     correlation = np.clip(correlation, -1.0, 1.0)
#     return correlation
