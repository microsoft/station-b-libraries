# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import pytest
import itertools
import torch
from torch.optim import SGD


import gp.numeric as numeric
from gp.moment_matching.pytorch.moment_matching_minimum import approximate_minimum as approx_min
from gp.moment_matching.pytorch.moment_matching_minimum_iterative import (
    approximate_minimum_with_prob_is_min as approx_min_iterative,
)


@pytest.mark.parametrize("size,num_points", itertools.product(range(1, 20), range(1, 15, 5)))
def test_calculate_cumulative_min_moments__same_in_two_versions(size: int, num_points: int):
    # Generate random Gaussian variables
    means, cov, _, _ = numeric.generate_random_variables(n_batch=num_points, size=size, mean_scale=0, mean_loc=1)
    means, cov = map(lambda x: torch.tensor(x), [means, cov])

    # Calculate the results using moment matching
    approx_mean, approx_var = approx_min(means=means, covariance=cov)
    approx_mean_iterative, approx_var_iterative, _ = approx_min_iterative(means=means, covariance=cov)

    approx_mean, approx_mean_iterative, approx_var, approx_var_iterative = map(
        lambda x: x.detach().numpy().ravel(), [approx_mean, approx_mean_iterative, approx_var, approx_var_iterative]
    )

    assert approx_mean == pytest.approx(approx_mean_iterative, rel=1e-7, abs=0)
    assert approx_var == pytest.approx(approx_var_iterative, rel=1e-7, abs=0)


@pytest.mark.parametrize("size,num_points", itertools.product(range(1, 20), range(1, 15, 5)))
def test_calculate_cumulative_min_moments_grads__same_in_two_versions(size: int, num_points: int):
    # Generate random Gaussian variables
    means, cov, _, _ = numeric.generate_random_variables(n_batch=num_points, size=size, mean_scale=0, mean_loc=1)
    means, cov = map(lambda x: torch.tensor(x, requires_grad=True), [means, cov])

    optimizer = SGD([means, cov], lr=1e-7)  #  Define a dummy optimizer to access .zero_grad()

    # Calculate the gradient wrt. to means and cov with method 1
    approx_mean, approx_var = approx_min(means=means, covariance=cov)
    objective = approx_mean.sum() + approx_var.sum()
    objective.backward()
    means_grad1, cov_grad1 = map(lambda x: x.grad.detach().numpy(), [means, cov])

    optimizer.zero_grad()

    # Calculate the gradient wrt. to means and cov with method 2
    approx_mean, approx_var, _ = approx_min_iterative(means=means, covariance=cov)
    objective = approx_mean.sum() + approx_var.sum()
    objective.backward()
    means_grad2, cov_grad2 = map(lambda x: x.grad.detach().numpy(), [means, cov])

    assert means_grad1 == pytest.approx(means_grad2, rel=1e-7, abs=0)
    assert cov_grad1 == pytest.approx(cov_grad2, rel=1e-7, abs=0)
