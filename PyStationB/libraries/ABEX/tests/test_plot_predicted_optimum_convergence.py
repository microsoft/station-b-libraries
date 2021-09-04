# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
import pytest
import yaml
from abex.settings import CustomDumper, OptimizerConfig
from abex.simulations.plot_predicted_optimum_convergence import (
    BATCH_COLUMN,
    _extract_iteration_number,
    load_optimum_file,
    load_experiment_label,
    load_experiment_df,
    validate_simulator_settings_same,
    load_config_from_expt_dir,
    load_combined_df,
)


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


@pytest.fixture
def zoomopt_config():
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
        "training": {"num_folds": 1, "iterations": 1, "upper_bound": 1, "plot_slice_1d": True, "plot_slice_2d": False},
        "zoomopt": {"batch": 6, "shrinking_factor": 0.6},
        "optimization_strategy": "Zoom",
    }
    return data


@pytest.fixture
def bayesopt_config_results_dir(tmp_path_factory, bayesopt_config):
    results_dir = tmp_path_factory.mktemp("bayesopt_config")
    bayesopt_config["results_dir"] = str(results_dir)

    config_path = results_dir / "config.yml"
    with open(str(config_path), "w+") as f_path:
        yaml.dump(bayesopt_config, f_path, Dumper=CustomDumper)

    # Add some data
    data_dir = results_dir / "fixed" / "seed0" / "iter1"
    data_dir.mkdir(exist_ok=True, parents=True)
    bayesopt_config["data_dir"] = str(data_dir)
    df = pd.DataFrame({"Ara": 1.0, "Crosstalk Ratio": 100.0}, index=[1])
    df.to_csv(data_dir / "optima.csv")
    return results_dir


@pytest.fixture
def zoomopt_config_results_dir(tmp_path_factory, zoomopt_config):
    results_dir = tmp_path_factory.mktemp("zoomopt_config")
    zoomopt_config["results_dir"] = str(results_dir)

    config_path = results_dir / "config.yml"
    with open(str(config_path), "w+") as f_path:
        yaml.dump(zoomopt_config, f_path, Dumper=CustomDumper)

    # Add some data
    data_dir = results_dir / "fixed" / "seed0" / "iter1"
    data_dir.mkdir(exist_ok=True, parents=True)
    zoomopt_config["data_dir"] = str(data_dir)
    df = pd.DataFrame({"Ara": 1.0, "Crosstalk Ratio": 100.0}, index=[1])
    df.to_csv(data_dir / "optima.csv")
    return results_dir


def test_extract_iteration_number() -> None:
    assert _extract_iteration_number(Path("iter1")) == 1
    assert _extract_iteration_number(Path("iter54321")) == 54321
    with pytest.raises(AttributeError):
        _extract_iteration_number("iter345")  # type: ignore
    with pytest.raises(ValueError):
        _extract_iteration_number(Path("xyz"))


def test_load_optimum_file(bayesopt_config_results_dir: Path) -> None:
    optimum_path = bayesopt_config_results_dir / "fixed" / "seed0" / "iter1" / "optima.csv"
    df = pd.read_csv(optimum_path)
    iteration_number = 1
    optimum_points = load_optimum_file(optimum_path, ["Ara"], iteration_number)
    assert "Ara" in optimum_points
    assert optimum_points["Ara"] == df["Ara"].values[0]  # type: ignore # auto
    assert optimum_points[BATCH_COLUMN] == iteration_number


def test_load_experiment_label(bayesopt_config_results_dir: Path, zoomopt_config_results_dir) -> None:
    bayesopt_config = load_config_from_expt_dir(bayesopt_config_results_dir, OptimizerConfig)
    assert load_experiment_label(bayesopt_config, []) == ""
    assert load_experiment_label(bayesopt_config, ["acquisition"]) == "MEAN_PLUGIN_EXPECTED_IMPROVEMENT"
    assert load_experiment_label(bayesopt_config, ["batch_strategy"]) == "LocalPenalization"
    assert (
        load_experiment_label(bayesopt_config, ["acquisition", "batch_strategy", "hmc"])
        == "MEAN_PLUGIN_EXPECTED_IMPROVEMENT LocalPenalization hmc"
    )
    assert load_experiment_label(bayesopt_config, ["optimization_strategy", "batch"]) == "Bayesian batch10"
    assert (
        load_experiment_label(bayesopt_config, ["acquisition", "batch_strategy"])
        == "MEAN_PLUGIN_EXPECTED_IMPROVEMENT LocalPenalization"
    )

    zoomopt_config = load_config_from_expt_dir(zoomopt_config_results_dir, OptimizerConfig)
    assert load_experiment_label(zoomopt_config, ["batch_strategy"]) == ""
    assert load_experiment_label(zoomopt_config, ["optimization_strategy", "batch"]) == "Zoom batch6"
    assert (
        load_experiment_label(zoomopt_config, ["optimization_strategy", "acquisition", "shrinking_factor"])
        == "Zoom  (0.6)"
    )


def test_load_experiment_df(bayesopt_config_results_dir: Path) -> None:
    df = load_experiment_df(bayesopt_config_results_dir, OptimizerConfig)
    with pytest.raises(Exception):
        load_experiment_df(Path("i-dont-exist"), OptimizerConfig)
    assert len(df) > 0


def test_load_combined_df(bayesopt_config_results_dir: Path) -> None:
    df = load_combined_df([bayesopt_config_results_dir], OptimizerConfig, ["acquisition", "batch_strategy"])
    assert len(df) > 0
    expt_df = pd.DataFrame(
        {
            "Ara": 1.0,
            "Batch Number": 1,
            "Sub-run Number": "seed0",
            "Experiment Name": "MEAN_PLUGIN_EXPECTED_IMPROVEMENT LocalPenalization",
        },
        index=[0],
    )
    # assert 'Ara' in df.columns
    # assert df['Experiment name'] == f"MEAN_PLUGIN_EXPECTED_IMPROVEMENT - LocalPenalization"
    assert df.equals(expt_df)


def test_validate_simulator_settings_same(bayesopt_config) -> None:
    good_config = OptimizerConfig(**bayesopt_config)
    validate_simulator_settings_same([good_config])

    with pytest.raises(Exception):
        validate_simulator_settings_same([])
