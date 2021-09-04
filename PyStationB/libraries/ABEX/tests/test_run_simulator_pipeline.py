# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import os
from argparse import Namespace
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import yaml
from abex.azure_config import AzureConfig
from abex.settings import CustomDumper, load_resolutions, OptimizerConfig
from azureml.pipeline.core import PipelineData
from abex.simulations.run_simulator_pipeline import AMLResources, specify_run_step, specify_plotting_step
from psbutils.misc import find_subrepo_directory

TESTS_DIR = find_subrepo_directory() / "tests"


@pytest.fixture
def dummy_azure_config(monkeypatch) -> AzureConfig:
    azure_dict = {
        "subscription_id": "abc123",
        "blob_storage": "blob123",
        "workspace_name": "ws123",
        "resource_group": "rg123",
        "aml_experiment": "expt123",
        "submit_to_aml": False,
        "compute_target": "dummy_compute_target",
    }
    return AzureConfig(**azure_dict)


@pytest.fixture
def mock_spec_path(tmp_path):
    spec_path = tmp_path / "specs"
    spec_path.mkdir(exist_ok=True)
    mock_spec = {
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
        "training": {"num_folds": 1, "iterations": 1, "upper_bound": 1, "plot_slice_1d": True, "plot_slice_2d": False},
        "bayesopt": {
            "batch": 5,
            "acquisition": ["@acquisition", "MEAN_PLUGIN_EXPECTED_IMPROVEMENT", "EXPECTED_IMPROVEMENT"],
            "batch_strategy": "LocalPenalization",
        },
        "zoomopt": {"batch": 5, "shrinking_factor": ["@shrinking_factor", 0.5, 0.7, 0.8]},
        "optimization_strategy": ["@optimization_strategy", "Bayesian", "Zoom"],
    }
    mock_spec_path = spec_path / "mock_spec_path.yml"
    with open(mock_spec_path, "w+") as f_path:
        yaml.dump(mock_spec, f_path, Dumper=CustomDumper)
    return mock_spec_path


@patch("abex.simulations.run_simulator_pipeline.Workspace")
@patch("abex.simulations.run_simulator_pipeline.ComputeTarget")
@patch("abex.simulations.run_simulator_pipeline.Datastore")
def get_aml_resources_object(
    mock_ws: MagicMock, mock_ct: MagicMock, mock_ds: MagicMock, dummy_azure_config: AzureConfig
) -> AMLResources:
    return AMLResources(dummy_azure_config)


def test_AMLResources(dummy_azure_config: AzureConfig) -> None:
    aml_resources = get_aml_resources_object(dummy_azure_config)
    assert hasattr(aml_resources, "compute_target")
    assert isinstance(aml_resources.compute_target, MagicMock)
    assert hasattr(aml_resources, "ws")
    assert hasattr(aml_resources, "env")
    assert hasattr(aml_resources, "datastore")


@patch("abex.simulations.run_simulator_pipeline.Environment")
def test_specify_conda_environment(dummy_azure_config: AzureConfig) -> None:
    aml_resources = get_aml_resources_object(dummy_azure_config)
    env = aml_resources.specify_conda_environment()
    assert isinstance(env, MagicMock)
    assert hasattr(env, "python.conda_dependencies")


@patch("abex.simulations.run_simulator_pipeline.PythonScriptStep")
@patch(
    "abex.simulations.run_simulator_pipeline.sys.argv",
    return_value=["path.py", "--spec_file", 0],
)
@pytest.mark.timeout(10)
def test_specify_run_step(dummy_script_step: MagicMock, dummy_azure_config: AzureConfig, mock_spec_path: Path) -> None:
    aml_resources = get_aml_resources_object(dummy_azure_config)
    arg_dict = {"spec_file": str(mock_spec_path), "base_seed": 1, "max_resolutions": 30, "num_runs": 1}
    args = Namespace(**arg_dict)
    parallel_steps, all_run_outputs, styled_subsets, temp_spec_paths = specify_run_step(
        args, aml_resources, Path("dummy_run_step"), OptimizerConfig, check_consistency=False  # type: ignore # auto
    )

    resolved_configs = load_resolutions(str(mock_spec_path))

    num_resolved_configs = len([x for x in resolved_configs])
    assert len(parallel_steps) == num_resolved_configs
    assert isinstance(parallel_steps[0], MagicMock)
    assert len(all_run_outputs) == num_resolved_configs
    assert isinstance(all_run_outputs[0], PipelineData)
    # styled subsets should be a dictionary
    assert isinstance(styled_subsets, dict)
    assert set([x for x in styled_subsets.values()][0]) == {
        "LocalPenalization - MEAN_PLUGIN_EXPECTED_IMPROVEMENT",
        "LocalPenalization - EXPECTED_IMPROVEMENT",
    }
    for spec_file in temp_spec_paths:
        os.remove(spec_file)


@patch("abex.simulations.run_simulator_pipeline.PythonScriptStep")
@patch(
    "abex.simulations.run_simulator_pipeline.sys.argv",
    return_value=["path.py", "--spec_file", 0],
)
@pytest.mark.timeout(10)
def test_specify_plotting_step(
    dummy_script_step: MagicMock, dummy_azure_config: AzureConfig, mock_spec_path: Path
) -> None:
    styled_subsets: Dict[str, List[str]] = {}
    experiment_labels: List[str] = []
    all_run_outputs: List[PipelineData] = []
    aml_resources: AMLResources = get_aml_resources_object(dummy_azure_config)
    plotting_step, output_plot = specify_plotting_step(
        styled_subsets, experiment_labels, all_run_outputs, aml_resources, Path("dummy_plotting_step")
    )
    assert isinstance(dummy_script_step, MagicMock)
    assert isinstance(plotting_step, MagicMock)
    assert isinstance(output_plot, PipelineData)
