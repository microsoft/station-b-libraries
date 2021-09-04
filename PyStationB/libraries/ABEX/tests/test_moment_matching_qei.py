# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats
from abex.emukit.moment_matching_qei import (
    calculate_cumulative_min_moments,
    correlation_from_covariance,
    get_next_cumulative_min_moments,
)


@pytest.mark.parametrize("size,num_points", itertools.product(range(1, 5), range(1, 10)))
def test_calculate_cumulative_min_moments__output_has_right_shape(size, num_points):
    #  Generate random Gaussian variables
    means = np.random.normal(size=[num_points, size], scale=0, loc=1)
    cov = np.stack([random_positive_definite_matrix(size) for _ in range(num_points)], axis=0)
    stds = np.sqrt(np.einsum("ijj->ij", cov))
    corr_matrix = correlation_from_covariance(cov, stds)
    # Calculate the results using moment matching
    approx_mean, approx_std, alphas = calculate_cumulative_min_moments(means, stds, corr_matrix)
    assert len(approx_mean) == size
    assert len(approx_std) == size
    assert len(alphas) == size
    for i in range(size):
        if i == 0:
            assert alphas[i] is None
        else:
            assert alphas[i].ndim == 1
            assert alphas[i].shape[0] == num_points
        assert approx_mean[i].ndim == 1
        assert approx_mean[i].shape[0] == num_points
        assert approx_std[i].ndim == 1
        assert approx_std[i].shape[0] == num_points


@pytest.mark.parametrize("size", range(1, 50, 5))
def test_calculate_cumulative_min_moments__output_is_finite(size):
    num_points = 20
    # Generate random Gaussian variables
    means = np.random.normal(size=[num_points, size], scale=0, loc=1)
    cov = np.stack([random_positive_definite_matrix(size) for _ in range(num_points)], axis=0)
    stds = np.sqrt(np.einsum("ijj->ij", cov))
    corr_matrix = correlation_from_covariance(cov, stds)
    # Calculate the results using moment matching
    approx_mean, approx_std, alphas = calculate_cumulative_min_moments(means, stds, corr_matrix)
    for i in range(size):
        if i == 0:
            assert alphas[i] is None
        else:
            assert np.isfinite(alphas[i]).all()
        assert np.isfinite(approx_mean[i]).all()
        assert np.isfinite(approx_std[i]).all()


@pytest.mark.timeout(30)
@pytest.mark.parametrize("mean_spread_scale,means_mean", [(1, 0), (1, -100), (1e3, 0), (1e-3, 0)])
def test_calculate_cumulative_min_moments__mean_std_exact_for_dim2(mean_spread_scale, means_mean):
    # Fix random seed
    np.random.seed(0)
    # Moment calculation is exact for two Gaussian variables => let size = 2
    size = 2
    num_points = 100
    # Generate random Gaussian variables
    means = np.random.normal(size=[num_points, size], scale=mean_spread_scale, loc=means_mean)
    cov = np.stack([random_positive_definite_matrix(size) for _ in range(num_points)], axis=0)
    stds = np.sqrt(np.einsum("ijj->ij", cov))
    corr_matrix = correlation_from_covariance(cov, stds)
    # Calculate the results using moment matching
    approx_mean, approx_std, alphas = calculate_cumulative_min_moments(means, stds, corr_matrix)
    assert np.isfinite(approx_mean[-1]).all()
    assert np.isfinite(approx_std[-1]).all()

    # Find MC estimates:
    num_samples = 10000
    num_repeats = 20
    error_prob = 1e-4

    for i in range(num_points):
        means_repeats, stds_repeats = np.zeros([num_repeats]), np.zeros([num_repeats])
        for j in range(num_repeats):
            theta_samples = np.random.multivariate_normal(mean=means[i], cov=cov[i], size=num_samples).min(axis=1)
            means_repeats[j] = theta_samples.mean()
            stds_repeats[j] = theta_samples.std()
        mc_mean = means_repeats.mean()
        mc_std = stds_repeats.mean()
        mc_mean_err = scipy.stats.norm.ppf(1 - (error_prob / 2)) * means_repeats.std()  # type: ignore # auto
        mc_std_err = scipy.stats.norm.ppf(1 - (error_prob / 2)) * stds_repeats.std()  # type: ignore # auto
        # Assert that estimates within confidence bound estimates
        assert pytest.approx(mc_mean, abs=mc_mean_err) == approx_mean[-1][i]
        assert pytest.approx(mc_std, abs=mc_std_err) == approx_std[-1][i]


@pytest.mark.timeout(20)
@pytest.mark.parametrize("mean_spread_scale,means_mean", [(1, 0), (1, -100), (1e3, 0), (1e-3, 0)])
def test_calculate_cumulative_min_moments__correlation_exact_for_dim3(mean_spread_scale, means_mean):
    # Fix random seed
    np.random.seed(0)
    # Moment calculation is exact for two Gaussian variables => let size = 2
    size = 3
    num_points = 100
    # Generate random Gaussian variables
    means = np.random.normal(size=[num_points, size], scale=mean_spread_scale, loc=means_mean)
    cov = np.stack([random_positive_definite_matrix(size) for _ in range(num_points)], axis=0)
    stds = np.sqrt(np.einsum("ijj->ij", cov))
    corr_matrix = correlation_from_covariance(cov, stds)
    # Calculate the results using moment matching for the first 2 variables
    theta_means, theta_stds, alphas = calculate_cumulative_min_moments(
        means[:, : size - 1], stds[:, : size - 1], corr_matrix[:, : size - 1, : size - 1]
    )
    assert np.isfinite(theta_means[-1]).all()
    assert np.isfinite(theta_stds[-1]).all()
    # Calculate the correlations to the 3rd variable
    last_output_idx = size - 1
    last_mean, last_std, last_output_theta_corr, _ = get_next_cumulative_min_moments(
        next_output_idx=last_output_idx,
        mean=means[:, last_output_idx],
        std=stds[:, last_output_idx],
        prev_stds=stds[:, :last_output_idx],
        corr_to_next=corr_matrix[:, :, last_output_idx],
        theta_means=theta_means,
        theta_stds=theta_stds,
        alphas=alphas,
    )

    # Find MC estimates:
    num_samples = 10000
    num_repeats = 20
    error_prob = 1e-4

    for i in range(num_points):
        corr_repeats = np.zeros([num_repeats])
        for j in range(num_repeats):
            y_samples = np.random.multivariate_normal(mean=means[i], cov=cov[i], size=num_samples)
            theta_prelast_samples = y_samples[:, :-1].min(axis=1)
            y_last_samples = y_samples[:, -1]
            corr_repeats[j] = np.corrcoef(y_last_samples, theta_prelast_samples)[0, 1]
        corr_mean = corr_repeats.mean()
        corr_err = scipy.stats.norm.ppf(1 - (error_prob / 2)) * corr_repeats.std()  # type: ignore # auto
        #  Assert that estimates within confidence bound estimates
        assert pytest.approx(corr_mean, abs=corr_err) == last_output_theta_corr[-1][i]
    return


def random_positive_definite_matrix(size: int) -> np.ndarray:
    factor = np.random.rand(size, size)
    mat: np.ndarray = factor.T @ factor
    return mat


def plot_mc_samples_vs_predicted_means_and_stds(
    size: int = 2,
    num_samples: int = 10000,
    num_repeats: int = 40,
    num_points: int = 100,
    confidence_interval: float = 0.9,
):
    means = np.random.normal(size=[num_points, size], scale=2.0, loc=0.0)
    cov = np.stack([random_positive_definite_matrix(size) for _ in range(num_points)], axis=0)
    stds = np.sqrt(np.einsum("ijj->ij", cov))
    corr_matrix = correlation_from_covariance(cov, stds)

    approx_mean, approx_std, alphas = calculate_cumulative_min_moments(means, stds, corr_matrix)
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
        mc_means_err[i] = scipy.stats.norm.ppf(1 - (error_prob / 2)) * means_repeats.std()  # type: ignore # auto
        mc_stds_err[i] = scipy.stats.norm.ppf(1 - (error_prob / 2)) * stds_repeats.std()  # type: ignore # auto

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
):
    assert size >= 3
    means = np.random.normal(size=[num_points, size], scale=2.0, loc=0.0)
    cov = np.stack([random_positive_definite_matrix(size) for _ in range(num_points)], axis=0)
    stds = np.sqrt(np.einsum("ijj->ij", cov))
    corr_matrix = correlation_from_covariance(cov, stds)

    # Calculate the results using moment matching for the first size-1 variables
    approx_mean, approx_std, alphas = calculate_cumulative_min_moments(
        means[:, : size - 1], stds[:, : size - 1], corr_matrix[:, : size - 1, : size - 1]
    )
    # Calculate the correlations to the 3rd variable
    last_mean, last_std, last_output_theta_corr, _ = get_next_cumulative_min_moments(
        next_output_idx=size - 1,
        mean=means[:, size - 1],
        std=stds[:, size - 1],
        prev_stds=stds[:, : size - 1],
        corr_to_next=corr_matrix[:, size - 1, :],
        theta_means=approx_mean,
        theta_stds=approx_std,
        alphas=alphas,
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
        mc_corrs_err[i] = scipy.stats.norm.ppf(1 - (error_prob / 2)) * corr_repeats.std()  # type: ignore # auto

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
