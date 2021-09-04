# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
import pytest

from abex.optimizers.optimizer_base import OptimizerBase
from psbutils.psblogging import logging_to_stdout

matplotlib.use("Agg")
# flake8: noqa: E402
from abex.settings import OptimizerConfig
from abex.data_settings import ParameterConfig, InputParameterConfig
from abex.simulations import DataLoop, SimulatedDataGenerator
from cellsig_sim.simulations import LatentCell


@pytest.fixture
def example_config():
    config = OptimizerConfig()
    config.data.inputs = {  # type: ignore # auto
        "LuxR": InputParameterConfig(lower_bound=1, upper_bound=5),
        "LasR": InputParameterConfig(lower_bound=1, upper_bound=5),
        "C6_on": InputParameterConfig(lower_bound=1, upper_bound=5),
        "C12_on": InputParameterConfig(lower_bound=1, upper_bound=5),
    }
    config.data.output_column = "Crosstalk Ratio"
    config.data.output_settings = ParameterConfig(normalise="None")
    config.bayesopt.batch = 3  # type: ignore # auto
    config.training.hmc = False  # Not testing HMC here, so get the speed-up from Max-likelihood
    config.training.num_folds = 0
    config.training.compute_optima = False
    config.seed = 42
    return config


@pytest.fixture
def temp_dir_config(tmp_path_factory, example_config: OptimizerConfig):
    results_dir: Path = tmp_path_factory.mktemp("Results-run_data_loop")
    data_dir: Path = tmp_path_factory.mktemp("Data-run_data_loop")
    example_config.data.folder = data_dir
    example_config.data.simulation_folder = data_dir
    example_config.data.files = {}
    example_config.results_dir = results_dir
    return example_config


@pytest.fixture
def init_batch():
    return np.array([[1, 2, 3, 2], [3, 4, 2, 3], [3, 0.5, 2, 1], [4, 3, 2, 4], [2, 3, 2, 3]])


class TestDataLoop:
    @pytest.mark.timeout(300)
    def test_simulated_data_loop_end_to_end(self, temp_dir_config, init_batch):
        logging_to_stdout()

        def sample_obj_side_effect(self, x):
            # Return an array of ones with same batch size as input
            assert x.ndim == 2
            assert x.shape[1] == 4
            return np.ones([x.shape[0], 1])

        # Mock the objective in simulator to return an array of 1s
        with mock.patch.object(LatentCell, "sample_objective", autospec=True, side_effect=sample_obj_side_effect):
            # Mock data generation of initial samples to return init_batch deterministically
            with mock.patch(
                "emukit.core.initial_designs.RandomDesign.get_samples", return_value=init_batch
            ) as mock_rd_get_samples:
                num_init_points = init_batch.shape[0]
                sim_loop = DataLoop.from_config(
                    data_generator=SimulatedDataGenerator(LatentCell()),
                    config=temp_dir_config,
                    num_init_points=num_init_points,
                )
                mock_rd_get_samples.assert_called_once_with(num_init_points)

            sim_loop.run_loop(1, optimizer=OptimizerBase.from_strategy(temp_dir_config))
        # There should be two data files: the generated initial batch and the 1 batch collected from loop
        assert len(sim_loop.data_settings.files) == 2
        # Check that the saved files are as expected
        init_batch_file = temp_dir_config.results_dir / sim_loop.initial_batch_filename
        batch_df = pd.read_csv(init_batch_file)
        np.testing.assert_array_equal(batch_df.to_numpy(), init_batch)  # type: ignore # auto

        # We use sim_loop.data_settings.files rather than temp_dir_config.data.files, because we expect only
        # the former to have been altered by running the loop.
        init_data_file = temp_dir_config.data.folder / sim_loop.data_settings.files[sim_loop.initial_data_key]
        init_data_df = pd.read_csv(init_data_file)
        assert len(init_data_df) == init_batch.shape[0]  # type: ignore # auto
        assert all(init_data_df[temp_dir_config.data.output_column] == 1)  # type: ignore # auto

        loop1_data_path = temp_dir_config.data.folder / sim_loop.data_settings.files[sim_loop.data_key_base.format(1)]
        loop1_data = pd.read_csv(loop1_data_path)
        assert len(loop1_data) == temp_dir_config.bayesopt.batch  # type: ignore # auto

        # Assert loop iteration-specific directory exists under results_dir
        assert sim_loop.iter_results_dir_base.format(1) in [
            str(subdir.name) for subdir in temp_dir_config.results_dir.iterdir() if subdir.is_dir()
        ]
