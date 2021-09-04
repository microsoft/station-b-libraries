# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import numpy as np
import pytest
from abex.simulations.toy_models import RadialFunction


class TestRadialFunction:
    def test_parameters_fixed(self):
        """Maximum and parameter space should be fixed and not change even if we toggle the log flag."""
        maximum = [1, 2, 3, 4]
        interval = (0.5, 100)

        for log_space in (True, False):
            rf = RadialFunction(maximum, interval, log_space=log_space)

            assert np.array_equal(maximum, rf.maximum)
            assert rf.parameter_space.get_bounds() == [interval for _ in maximum]

    def test_maximum_not_in_parameter_space(self):
        maximum = [1, 2, 3, 4]
        bounds = (0.5, 2)
        with pytest.raises(ValueError):
            RadialFunction(maximum, bounds, log_space=True)

    def test_sampling(self):
        X = np.array(
            [
                [1, 2],
                [0, 0],
                [3, 1],
            ]
        )
        maximum = np.array([0, 0])

        objective_expected = np.array(
            [
                [-5],
                [0],
                [-10],
            ]
        )

        bounds = (-5, 5)

        rf_linear = RadialFunction(maximum, bounds, log_space=False)  # type: ignore # auto
        assert np.array_equal(rf_linear.sample_objective(X), objective_expected)

        rf_log = RadialFunction(10 ** maximum, (1e-5, 1e5), log_space=True)  # type: ignore
        assert np.array_equal(rf_log.sample_objective(10 ** X), objective_expected)
