# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import pytest
import scipy.stats
import numpy as np
import itertools
import torch


import gp.numeric as numeric
from gp.moment_matching.pytorch.moment_matching_minimum_iterative import (
    get_next_cumulative_min_moments,
    calculate_cumulative_min_moments,
)


@pytest.mark.parametrize("size,num_points", itertools.product(range(1, 5), range(1, 10)))
def test_calculate_cumulative_min_moments__output_has_right_shape(size: int, num_points: int):
    # Generate random Gaussian variables
    means, cov, _, _ = numeric.generate_random_variables(n_batch=num_points, size=size, mean_scale=0, mean_loc=1)
    means, cov = map(lambda x: torch.tensor(x), [means, cov])

    # Calculate the results using moment matching
    approx_means, approx_vars, prob_is_mins = calculate_cumulative_min_moments(
        means=means,
        covariance=cov,
    )
    assert len(approx_means) == size
    assert len(approx_vars) == size
    assert len(prob_is_mins) == size

    for i in range(size):
        assert approx_means[i].ndim == 1
        assert approx_means[i].shape[0] == num_points
        assert approx_vars[i].ndim == 1
        assert approx_vars[i].shape[0] == num_points

        if i >= 1:
            assert prob_is_mins[i].ndim == 2
            assert prob_is_mins[i].shape[0] == num_points
            assert prob_is_mins[i].shape[1] == i + 1


@pytest.mark.parametrize("size", range(1, 50))
def test_calculate_cumulative_min_moments__output_is_finite(size: int):
    num_points = 20

    # Generate random Gaussian variables
    means, cov, _, _ = numeric.generate_random_variables(n_batch=num_points, size=size, mean_scale=0, mean_loc=1)
    means, cov = map(lambda x: torch.tensor(x), [means, cov])

    # Calculate the results using moment matching
    approx_means, approx_vars, prob_is_mins = calculate_cumulative_min_moments(
        means=means,
        covariance=cov,
    )

    for i in range(size):
        assert torch.isfinite(approx_means[i]).all()
        assert torch.isfinite(approx_vars[i]).all()

        if i >= 1:
            assert torch.isfinite(prob_is_mins[i]).all()


@pytest.mark.parametrize("mean_spread_scale,means_mean", [(1, 0), (1, -100), (1e3, 0), (1e-3, 0)])
def test_calculate_cumulative_min_moments__mean_std_exact_for_dim2(
    mean_spread_scale, means_mean, num_mc_samples=10000, mc_pvalue_bound=0.005, num_points=100
):
    # Fix random seed
    np.random.seed(0)
    # Moment calculation is exact for two Gaussian variables => let size = 2
    size = 2

    # Generate random Gaussian variables
    means, cov, _, _ = numeric.generate_random_variables(
        n_batch=num_points, size=size, mean_scale=mean_spread_scale, mean_loc=means_mean
    )
    means, cov = map(lambda x: torch.tensor(x), [means, cov])
    # Calculate the results using moment matching
    approx_means, approx_vars, prob_is_mins = calculate_cumulative_min_moments(
        means=means,
        covariance=cov,
    )
    assert torch.isfinite(approx_means[-1]).all()
    assert torch.isfinite(approx_vars[-1]).all()

    # Find MC estimates:
    for i in range(num_points):
        exact_alg_mean = approx_means[-1][i].numpy()
        exact_alg_var = approx_vars[-1][i].numpy()
        exact_alg_std = np.sqrt(exact_alg_var)

        theta_samples = np.random.multivariate_normal(mean=means[i], cov=cov[i], size=num_mc_samples).min(axis=1)

        _, pvalue_mean = scipy.stats.ttest_1samp(theta_samples, popmean=exact_alg_mean)
        # Validate that we can't reject the null hypothesis (means equal)
        # An allowed p-value of 0.01 means that this test is expected to fail 1 / 100 times
        assert pvalue_mean > mc_pvalue_bound

        var_samples = (theta_samples - exact_alg_mean) ** 2
        _, pvalue_var = scipy.stats.ttest_1samp(var_samples, popmean=exact_alg_std ** 2)
        # Validate that we can't reject the null hypothesis (standard deviations equal)
        assert pvalue_var > mc_pvalue_bound
    return


@pytest.mark.parametrize("mean_spread_scale,means_mean", [(1, 0), (1, -100), (1e3, 0), (1e-3, 0)])
def test_calculate_cumulative_min_moments__correlation_exact_for_dim3(
    mean_spread_scale, means_mean, num_mc_samples=10000, mc_pvalue_bound=0.005, num_points=100
):
    # Fix random seed
    np.random.seed(0)
    size = 3  # For three variables the calculation of the correlation coefficient is exact

    # Generate random Gaussian variables
    means, cov, _, _ = numeric.generate_random_variables(
        n_batch=num_points, size=size, mean_scale=mean_spread_scale, mean_loc=means_mean
    )

    means, cov = map(lambda x: torch.tensor(x), [means, cov])
    # Calculate the results using moment matching
    cum_min_means, cum_min_vars, prob_is_mins = calculate_cumulative_min_moments(
        means=means[..., : size - 1],
        covariance=cov[..., : size - 1, : size - 1],
    )

    assert torch.isfinite(cum_min_means[-1]).all()
    assert torch.isfinite(cum_min_vars[-1]).all()
    # Calculate the correlations to the 3rd variable
    last_output_idx = size - 1
    last_mean, last_var, last_output_cum_min_cov, *_ = get_next_cumulative_min_moments(
        mean=means[..., last_output_idx],
        variance=cov[..., last_output_idx, last_output_idx],
        cov_to_next=cov[..., :last_output_idx, last_output_idx],
        cum_min_mean=cum_min_means[-1],
        cum_min_var=cum_min_vars[-1],
        prob_is_min=prob_is_mins[-1],
    )

    for i in range(num_points):
        exact_alg_cov = last_output_cum_min_cov[i].numpy()

        y_samples = np.random.multivariate_normal(mean=means[i], cov=cov[i], size=num_mc_samples)
        # Get the minimum of all Gaussians excluding the last one
        theta_prelast_samples = y_samples[:, :-1].min(axis=1)
        # Extract samples from the last Gaussian
        y_last_samples = y_samples[:, -1]
        # Calculated the correlation between last Gaussian and the minimum of previous Gaussians
        empirical_cov_samples = (y_last_samples - means[i, -1].numpy()) * (
            theta_prelast_samples - cum_min_means[-1][i].numpy()
        )
        _, pvalue_corr = scipy.stats.ttest_1samp(empirical_cov_samples, popmean=exact_alg_cov)
        # Validate that we can't reject the null hypothesis (correlations equal)
        # An allowed p-value of 0.01 means that this test is expected to fail 1 / 100 times
        assert pvalue_corr > mc_pvalue_bound


@pytest.mark.parametrize("mean_spread_scale", [1e-1, 1, 10])
@pytest.mark.parametrize("num_gaussians", [3, 5, 10, 15])
def test_calculate_cumulative_min_moments__approximation_close_to_true(mean_spread_scale, num_gaussians):
    # Fix random seed
    np.random.seed(0)
    # Moment calculation is exact for two Gaussian variables => let size = 2
    num_points = 100
    # Generate random Gaussian variables
    means, cov, _, _ = numeric.generate_random_variables(
        n_batch=num_points, size=num_gaussians, mean_scale=mean_spread_scale, mean_loc=0
    )
    means, cov = map(lambda x: torch.tensor(x), [means, cov])
    # Calculate the results using moment matching
    approx_means, approx_vars, *_ = calculate_cumulative_min_moments(
        means=means,
        covariance=cov,
    )

    assert torch.isfinite(approx_means[-1]).all()
    assert torch.isfinite(approx_vars[-1]).all()

    # -- Find Monte Carlo estimates.
    num_samples = 10000
    mc_means, mc_stds = numeric.monte_carlo_minimum_estimate(means=means, covariances=cov, n_samples=num_samples)

    # Assert that estimates correlate quite highly with the true values
    assert np.corrcoef(mc_means, approx_means[-1])[0, 1] > 0.9
    assert np.corrcoef(mc_stds, approx_vars[-1])[0, 1] > 0.9
