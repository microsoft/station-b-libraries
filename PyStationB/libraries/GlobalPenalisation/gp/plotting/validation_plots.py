# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import gp.moment_matching.pytorch.moment_matching_minimum as gradients
from gp.numeric import batch_correlation_from_covariance, random_positive_definite_matrix


def _prepare_variables(
    size: int, num_points: int, scale: float = 2.0, loc: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function prepares random variables.

    Returns:
        means: means matrix, shape (num_points, size)
        cov: covariance matrices, shape (num_points, size, size)
        stds: standard deviations matrix, shape (num_points, size)
        corr_matrix: correlation matrices, shape (num_points, size, size)
    """
    means = np.random.normal(size=(num_points, size), scale=scale, loc=loc)
    cov = np.stack([random_positive_definite_matrix(size) for _ in range(num_points)], axis=0)
    stds = np.sqrt(np.einsum("ijj->ij", cov))
    corr_matrix = batch_correlation_from_covariance(cov, stds)

    return means, cov, stds, corr_matrix


def plot_mc_samples_vs_predicted_means_and_stds(
    size: int = 2,
    num_samples: int = 10000,
    num_repeats: int = 40,
    num_points: int = 100,
    confidence_interval: float = 0.9,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    means, cov, stds, corr_matrix = _prepare_variables(size=size, num_points=num_points)

    approx_mean, approx_std, alpha_pdfs, alpha_cdfs = gradients.calculate_cumulative_min_moments(
        means, stds, corr_matrix
    )
    # Find MC estimates:
    error_prob = 1.0 - confidence_interval
    assert error_prob > 0

    mc_means = np.zeros([num_points])
    mc_stds = np.zeros_like(mc_means)
    mc_means_err = np.zeros_like(mc_means)
    mc_stds_err = np.zeros_like(mc_stds)
    for i in range(num_points):
        means_repeats, stds_repeats = np.zeros([num_repeats]), np.zeros([num_repeats])
        for j in range(num_repeats):
            theta_samples = np.random.multivariate_normal(mean=means[i], cov=cov[i], size=num_samples).min(axis=1)
            means_repeats[j] = theta_samples.mean()
            stds_repeats[j] = theta_samples.std()
        mc_means[i] = means_repeats.mean()
        mc_stds[i] = stds_repeats.mean()
        mc_means_err[i] = scipy.stats.norm.ppf(1 - (error_prob / 2)) * means_repeats.std()
        mc_stds_err[i] = scipy.stats.norm.ppf(1 - (error_prob / 2)) * stds_repeats.std()

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    xgrid = np.linspace(mc_means.min(), mc_means.max(), 100)
    ax1.plot(xgrid, xgrid, alpha=0.5, color="red")
    ax1.errorbar(
        approx_mean[-1], mc_means, yerr=mc_means_err * 10, fmt="o", color="black", ecolor="black", elinewidth=0.4
    )
    ax1.set_xlabel("Predicted mean (Moment matching)")
    ax1.set_ylabel("Monte Carlo estimate of mean")

    stdgrid = np.linspace(0, mc_stds.max(), 100)
    ax2.plot(stdgrid, stdgrid, alpha=0.5, color="blue")
    ax2.errorbar(approx_std[-1], mc_stds, yerr=mc_stds_err * 10, fmt="o", color="black", ecolor="black", elinewidth=0.4)
    ax1.set_xlabel("Predicted st. deviation (Moment matching)")
    ax1.set_ylabel("Monte Carlo estimate of st. deviation")
    return fig, (ax1, ax2)


def plot_mc_samples_vs_predicted_correlations(
    size: int = 3,
    num_samples: int = 10000,
    num_repeats: int = 40,
    num_points: int = 100,
    confidence_interval: float = 0.9,
) -> Tuple[plt.Figure, plt.Axes]:
    assert size >= 3, "Size needs to be at least 3."
    means, cov, stds, corr_matrix = _prepare_variables(size=size, num_points=num_points)

    # Calculate the results using moment matching for the first size-1 variables
    approx_mean, approx_std, alpha_pdfs, alpha_cdfs = gradients.calculate_cumulative_min_moments(
        means[:, : size - 1], stds[:, : size - 1], corr_matrix[:, : size - 1, : size - 1]
    )
    # Calculate the correlations to the 3rd variable
    last_mean, last_std, last_output_theta_corr, *_ = gradients.get_next_cumulative_min_moments(
        next_output_idx=size - 1,
        mean=means[:, size - 1],
        std=stds[:, size - 1],
        prev_stds=stds[:, : size - 1],
        corr_to_next=corr_matrix[:, size - 1, :],
        theta_means=approx_mean,
        theta_stds=approx_std,
        alpha_pdfs=alpha_pdfs,
        alpha_cdfs=alpha_cdfs,
    )
    # Find MC estimates:
    error_prob = 1.0 - confidence_interval
    assert error_prob > 0

    mc_corrs = np.zeros([num_points])
    mc_corrs_err = np.zeros_like(mc_corrs)
    for i in range(num_points):
        corr_repeats = np.zeros([num_repeats])
        for j in range(num_repeats):
            y_samples = np.random.multivariate_normal(mean=means[i], cov=cov[i], size=num_samples)
            theta_prelast_samples = y_samples[:, :-1].min(axis=1)
            y_last_samples = y_samples[:, -1]
            corr_repeats[j] = np.corrcoef(y_last_samples, theta_prelast_samples)[0, 1]
        mc_corrs[i] = corr_repeats.mean()
        mc_corrs_err[i] = scipy.stats.norm.ppf(1 - (error_prob / 2)) * corr_repeats.std()

    fig, ax = plt.subplots(figsize=(6, 6))
    xgrid = np.linspace(mc_corrs.min(), mc_corrs.max(), 100)
    ax.plot(xgrid, xgrid, alpha=0.5, color="red")
    ax.errorbar(
        last_output_theta_corr[-1],
        mc_corrs,
        yerr=mc_corrs_err * 10,
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=0.4,
    )
    ax.set_xlabel("Predicted Correlation (Moment matching)")
    ax.set_ylabel("Monte Carlo estimate of correlation")
    return fig, ax
