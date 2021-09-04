# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import collections
from pathlib import Path
from typing import Sequence, Tuple

import abex.data_settings
import numpy as np
import pandas as pd
import pytest
from abex import settings
from abex.optimizers.optimizer_base import OptimizerBase
from abex.simulations import DataLoop, SimulatedDataGenerator
from abex.simulations.toy_models import RadialFunction
from psbutils.psblogging import logging_to_stdout


@pytest.fixture(scope="module")
def bounds() -> Tuple[float, float]:
    return 1, 50


@pytest.fixture(scope="module")
def maximum() -> Sequence[float]:
    return 7, 14, 20


@pytest.fixture(scope="module")
def objective_name() -> str:
    return "Objective"


@pytest.fixture(scope="module")
def example_config(
    bounds: Tuple[float, float], maximum: Sequence[float], objective_name: str
) -> settings.OptimizerConfig:
    config = settings.OptimizerConfig()
    config.data.inputs = collections.OrderedDict(
        [
            (f"x{i}", abex.data_settings.InputParameterConfig(lower_bound=bounds[0], upper_bound=bounds[1]))
            for i, _ in enumerate(maximum, 1)
        ]
    )
    config.data.output_column = objective_name
    config.data.output_settings = abex.data_settings.ParameterConfig(
        normalise=abex.data_settings.NormalisationType.NONE
    )

    config.zoomopt = settings.ZoomOptSettings(batch=3 ** len(maximum), shrinking_factor=0.7 ** 3)

    config.training.num_folds = 1
    config.training.hmc = False
    config.seed = 42
    return config


@pytest.fixture(scope="module")
def temp_dir_zoomopt_config(tmp_path_factory, example_config: settings.OptimizerConfig) -> settings.OptimizerConfig:
    results_dir: Path = tmp_path_factory.mktemp("Results-toy_model_optimization-grid")
    data_dir: Path = tmp_path_factory.mktemp("Data-toy_model_optimization-grid")
    example_config.data.folder = data_dir
    example_config.data.files = {}
    example_config.results_dir = results_dir
    example_config.optimization_strategy = settings.OptimizationStrategy.ZOOM
    return example_config


@pytest.mark.timeout(10)
def test_optimize_toy_function_using_zoom(
    bounds: Tuple[float, float], maximum: Sequence[float], temp_dir_zoomopt_config, objective_name: str
):
    logging_to_stdout()
    n_iterations: int = 10

    simulator = RadialFunction(maximum, bounds, log_space=True)
    data_generator = SimulatedDataGenerator(simulator, objective_col_name=objective_name)

    assert temp_dir_zoomopt_config.zoomopt is not None  # for mypy
    loop = DataLoop.from_config(
        data_generator=data_generator,
        config=temp_dir_zoomopt_config,
        num_init_points=temp_dir_zoomopt_config.zoomopt.batch,
    )

    # Run the optimization
    loop.run_loop(n_iterations, optimizer=OptimizerBase.from_strategy(temp_dir_zoomopt_config))

    # At the last iteration we expect that
    #    (a) all points are close to each other (the cube is very small)
    #    (b) we found a maximum
    # Let's evaluate objective at each row, then
    last_iter_dir: Path = temp_dir_zoomopt_config.results_dir / loop.iter_results_dir_base.format(n_iterations - 1)
    last_iterations = pd.read_csv(last_iter_dir / temp_dir_zoomopt_config.experiment_batch_path.name)

    for i, row in last_iterations.iterrows():  # type: ignore # auto
        maximum_found: Sequence[float] = [row[f"x{i}"] for i, _ in enumerate(maximum, 1)]  # type: ignore # auto

        # Evaluate the objective at that point
        X = np.array([maximum_found, maximum])
        Y = simulator.sample_objective(X).ravel()
        objective_at_found_maximum, objective_at_maximum = Y[0], Y[1]
        # Objective at true maximum should be 0 for this function. Hence, we check if objective at 'found' optimum
        # is less than 0.02.
        assert abs(objective_at_found_maximum - objective_at_maximum) < 0.02, (
            f"Objective at {i}th was " f"{objective_at_found_maximum}"
        )
