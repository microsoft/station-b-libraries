# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import pytest

# import numpy as np
# import torch

# from gp.moment_matching.pytorch.moment_matching_minimum import calculate_cumulative_min_moments
# import gp.numeric as numeric


@pytest.mark.skip("Test not finished")
@pytest.mark.parametrize("mean_spread_scale", [1e-1, 1, 10])
@pytest.mark.parametrize("num_gaussians", [3, 5, 10])
@pytest.mark.parametrize("batch_size", [3, 7, 100])
def test_calculate_cumulative_min_moments__gradients_correct(mean_spread_scale, num_gaussians, batch_size):
    pass
    # Fix random seed
    # np.random.seed(42)
    # Fixed step for computing numerical derivatives
    # TODO: This is not used now, when we finish it we should remove noqa
    # epsilon = 1e-7  # noqa: F841
    # # Moment calculation is exact for two Gaussian variables => let size = 2

    # # Generate random Gaussian variables
    # means, cov, stds, corr_matrix = numeric.generate_random_variables(
    #     n_batch=batch_size, size=num_gaussians, mean_scale=mean_spread_scale, mean_loc=0
    # )
    # means, stds, corr_matrix = map(lambda x: torch.tensor(x), [means, stds, corr_matrix])

    # # Calculate the results using moment matching
    # approx_mean, approx_std, alpha_pdfs, alpha_cdfs = calculate_cumulative_min_moments(means, stds, corr_matrix)
    # cumulative_min_mean = approx_mean[-1]
    # cumulative_min_std = approx_std[-1]  # noqa: F841
    # # Get the pytorch gradient
    # cumulative_min_mean.sum().backward()
    # for i in range(num_gaussians):
    #     # Get numerical gradient
    #     shifted_means = means.copy()  # noqa: F841

    # # -- Find Monte Carlo estimates.
    # num_samples = 10000  # The number of MC samples.
    # # Both arrays have shape (batch_size,).
    # mc_means, mc_stds = numeric.monte_carlo_minimum_estimate(means=means, covariances=cov, n_samples=num_samples)

    # # Assert that estimates correlate quite highly with the true values
    # assert np.corrcoef(mc_means, approx_mean[-1])[0, 1] > 0.9
    # assert np.corrcoef(mc_stds, approx_std[-1])[0, 1] > 0.9
