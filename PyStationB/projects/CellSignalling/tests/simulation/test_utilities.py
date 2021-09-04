# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import numpy as np
import pytest
from cellsig_sim.simulations import lognormal_from_mean_and_std


@pytest.mark.parametrize("mean, std", [(1, 1), (1, 0.1), (0.1, 1e-4), (1e3, 1.0), (1e3, 20.0)])
def test_lognormal_sampling__gives_desired_mean_and_std(mean, std):
    np.random.seed(5)
    samples = lognormal_from_mean_and_std(mean=mean, std=std, size=100000)
    assert samples.mean() == pytest.approx(mean, rel=1e-2)
    assert samples.std() == pytest.approx(std, rel=1e-1)
