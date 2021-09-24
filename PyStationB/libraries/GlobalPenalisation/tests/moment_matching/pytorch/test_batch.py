# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Test that the invocation of the APIs in batch-mode gives same results as just
calling multiple times.
"""
import pytest
import torch


import gp.numeric as numeric
from gp.moment_matching.pytorch.moment_matching_minimum import approximate_minimum
from gp.moment_matching.pytorch.moment_matching_minimum_iterative import (
    approximate_minimum_with_prob_is_min,
)


@pytest.mark.parametrize(
    "size, batch_size",
    [
        (5, 5),
        (1, 4),
        (4, 1),
        (3, 2),
        (1, 1),
    ],
)
def test_approximate_minimum__same_in_batch_mode(size: int, batch_size: int):
    means, cov, _, _ = numeric.generate_random_variables(n_batch=batch_size, size=size, mean_scale=0, mean_loc=1)
    means, cov = map(lambda x: torch.tensor(x), [means, cov])

    # Calculate sequentially:
    approx_means, approx_vars = [], []
    for i in range(batch_size):
        approx_mean, approx_var = approximate_minimum(means=means[i], covariance=cov[i])
        approx_means.append(approx_mean)
        approx_vars.append(approx_var)
    approx_mean_seq = torch.stack(approx_means, dim=0)
    approx_var_seq = torch.stack(approx_vars, dim=0)
    # Calculate in batch
    approx_mean_batch, approx_var_batch = approximate_minimum(means=means, covariance=cov)
    assert torch.all(approx_mean_seq == approx_mean_batch)
    assert torch.all(approx_var_seq == approx_var_batch)


@pytest.mark.parametrize(
    "size, batch_size",
    [
        (5, 5),
        (1, 4),
        (4, 1),
        (3, 2),
        (1, 1),
    ],
)
def test_approximate_minimum_with_prob_is_min__same_in_batch_mode(size: int, batch_size: int):
    means, cov, _, _ = numeric.generate_random_variables(n_batch=batch_size, size=size, mean_scale=0, mean_loc=1)
    means, cov = map(lambda x: torch.tensor(x), [means, cov])

    # Calculate sequentially:
    approx_means, approx_vars, prob_is_mins = [], [], []
    for i in range(batch_size):
        approx_mean, approx_var, prob_is_min = approximate_minimum_with_prob_is_min(means=means[i], covariance=cov[i])
        approx_means.append(approx_mean)
        approx_vars.append(approx_var)
        prob_is_mins.append(prob_is_min)
    approx_mean_seq = torch.stack(approx_means, dim=0)
    approx_var_seq = torch.stack(approx_vars, dim=0)
    prob_is_min_seq = torch.stack(prob_is_mins, dim=0)
    # Calculate in batch
    approx_mean_batch, approx_var_batch, prob_is_min_batch = approximate_minimum_with_prob_is_min(
        means=means, covariance=cov
    )
    assert torch.all(approx_mean_seq == approx_mean_batch)
    assert torch.all(approx_var_seq == approx_var_batch)
    assert torch.all(prob_is_min_seq == prob_is_min_batch)
