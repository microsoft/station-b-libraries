# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import pytest
from abex.simulations.plot_predicted_optimum_convergence import (
    validate_simulator_settings_same,
)
from cellsig_sim.optconfig import CellSignallingOptimizerConfig


@pytest.fixture
def bayesopt_config():
    data = {
        "data": {
            "folder": "DummyFolder",
            "inputs": {
                "Ara": {  # mM
                    "unit": "mM",
                    "log_transform": True,
                    "lower_bound": 0.01,
                    "upper_bound": 100.0,
                    "normalise": "Full",
                }
            },
            "output_column": "Crosstalk Ratio",
            "output_settings": {"log_transform": True, "normalise": "Full"},
        },
        "model": {"kernel": "RBF", "add_bias": True},
        "training": {
            "hmc": True,
            "num_folds": 1,
            "iterations": 1,
            "upper_bound": 1,
            "plot_slice_1d": True,
            "plot_slice_2d": False,
        },
        "bayesopt": {
            "batch": 10,
            "acquisition": "MEAN_PLUGIN_EXPECTED_IMPROVEMENT",
            "batch_strategy": "LocalPenalization",
        },
        "optimization_strategy": "Bayesian",
    }
    return data


def test_validate_simulator_settings_same(bayesopt_config) -> None:
    good_config = CellSignallingOptimizerConfig(**bayesopt_config)
    validate_simulator_settings_same([good_config])
    with pytest.raises(Exception):
        validate_simulator_settings_same([])
