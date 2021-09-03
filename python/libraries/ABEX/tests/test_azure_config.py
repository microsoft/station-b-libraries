# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from typing import List

import pytest
from abex.azure_config import copy_relevant_script_args
from abex.scripts.run import RunConfig
from abex.settings import OptimizationStrategy


def mock_run_config_with_enum():
    return RunConfig(
        num_runs=2,
        base_seed=123,
        strategy=OptimizationStrategy.ZOOM,
    )


def mock_run_config_with_submit_to_aml():
    return RunConfig(num_runs=2, base_seed=123, submit_to_aml=True)


def mock_run_config_with_true():
    return RunConfig(num_runs=2, base_seed=123, enable_multiprocessing=True)


def mock_run_config_with_false():
    return RunConfig(num_runs=2, base_seed=123, enable_multiprocessing=False)


def mock_run_config_with_none():
    return RunConfig(num_runs=2, base_seed=123, spec_name=None)


def mock_run_config_with_default():
    return RunConfig(num_runs=10, base_seed=123)


@pytest.mark.parametrize(
    "run_config,expected_args",
    [
        (mock_run_config_with_enum(), ["--base_seed", 123, "--num_runs", 2]),
        (mock_run_config_with_submit_to_aml(), ["--base_seed", 123, "--num_runs", 2]),
        (mock_run_config_with_true(), ["--base_seed", 123, "--enable_multiprocessing", "--num_runs", 2]),
        (mock_run_config_with_false(), ["--base_seed", 123, "--num_runs", 2]),
        (mock_run_config_with_none(), ["--base_seed", 123, "--num_runs", 2]),
        (mock_run_config_with_default(), ["--base_seed", 123]),
    ],
)
def test_copy_relevant_script_args(run_config: RunConfig, expected_args: List[str]):
    """
    Assert that script_args (to be passed when calling script in AML) is created correctly under
    a number of different scenarios.
    """
    script_args_enum = copy_relevant_script_args(run_config)
    assert script_args_enum == expected_args
