# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import collections
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
from abex.optimizers.optimizer_base import OptimizerBase
from abex.settings import NonNegativeInt, OptimizationStrategy, OptimizerConfig
from abex.data_settings import ParameterConfig, InputParameterConfig
from abex.simulations import DataLoop, SimulatedDataGenerator
from abex.simulations.toy_models import RadialFunction


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
def example_config(bounds: Tuple[float, float], maximum: Sequence[float], objective_name: str) -> OptimizerConfig:
    config = OptimizerConfig()
    config.data.inputs = collections.OrderedDict(
        [
            (f"x{i}", InputParameterConfig(lower_bound=bounds[0], upper_bound=bounds[1], normalise="Full"))
            for i, _ in enumerate(maximum, 1)
        ]
    )
    config.data.output_column = objective_name
    config.data.output_settings = ParameterConfig(normalise="None")
    config.bayesopt.batch = NonNegativeInt(5)
    config.model.add_bias = True

    config.training.num_folds = 1
    config.training.plot_slice1d = False
    config.training.plot_slice2d = False
    config.training.num_folds = 1
    config.training.hmc = False
    config.seed = 42
    config.optimization_strategy = OptimizationStrategy.BAYESIAN
    return config


@pytest.fixture(scope="module")
def temp_dir_config(tmp_path_factory, example_config: OptimizerConfig) -> OptimizerConfig:
    results_dir: Path = tmp_path_factory.mktemp("Results-toy_model_optimization")
    data_dir: Path = tmp_path_factory.mktemp("Data-toy_model_optimization")
    example_config.data.folder = data_dir
    example_config.data.files = {}
    example_config.results_dir = results_dir
    return example_config


# @pytest.mark.timeout(1800)
@pytest.mark.skip("Can cause ADO timeout on both Windows and Linux")
def test_optimize_toy_function(
    bounds: Tuple[float, float], maximum: Sequence[float], temp_dir_config: OptimizerConfig, objective_name: str
):
    n_iterations: int = 3

    simulator = RadialFunction(maximum, bounds, log_space=True)
    data_generator = SimulatedDataGenerator(simulator, objective_col_name=objective_name)

    loop = DataLoop.from_config(data_generator=data_generator, config=temp_dir_config, num_init_points=40)

    # Run the optimization
    loop.run_loop(n_iterations, optimizer=OptimizerBase.from_strategy(temp_dir_config))

    # Retrieve the optimum
    found_optima: Path = temp_dir_config.results_dir / loop.iter_results_dir_base.format(n_iterations) / "optima.csv"
    df: pd.DataFrame = pd.read_csv(found_optima)  # type: ignore # auto
    row = df.iloc[0]
    maximum_found: Sequence[float] = [row[f"x{i}"] for i, _ in enumerate(maximum, 1)]

    # Evaluate the objective at that point
    X = np.array([maximum_found, maximum])
    Y = simulator.sample_objective(X).ravel()
    objective_at_found_maximum, objective_at_maximum = Y[0], Y[1]
    # Objective at true maximum should be 0 for this function. Hence, we check if objective at 'found' optimum
    # is less than 0.02 (temporarily 0.061 because we need a prior - see bug 19103)
    # Note: since n_iterations is only = 3, this assertion is very sensitive to any small changes in configuration
    assert abs(objective_at_found_maximum - objective_at_maximum) < 0.061
