# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import abex.data_settings
import abex.settings as settings
import pydantic
import pytest
import yaml
from psbutils.misc import find_subrepo_directory

SUBREPO_DIR = find_subrepo_directory()


@pytest.fixture(scope="module")
def mock_spec_filename() -> Generator:
    mock_spec_name = "mock_spec.yml"
    spec_path = SUBREPO_DIR / "tests" / "data" / "Specs" / mock_spec_name
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
        "bayesopt": {"batch": 5},
    }
    with open(spec_path, "w+") as f_path:
        yaml.dump(data, f_path, default_flow_style=False, Dumper=settings.CustomDumper)
    yield spec_path
    os.remove(spec_path)


def test_load_resolutions(mock_spec_filename: str) -> None:
    """Assert that given a config name, it is located and parsed correctly"""
    resolution_generator: Generator[
        List[Tuple[Dict[str, Any], settings.OptimizerConfig]], None, None
    ] = settings.load_resolutions(mock_spec_filename)
    _, config = next(resolution_generator)[0]
    assert config.resolution_spec == ""
    assert config.bayesopt.batch == 5
    assert config.data.folder == Path("DummyFolder")
    assert config.data.simulation_folder == Path("DummyFolder/fixed/seed0")
    assert config.results_dir == Path("Results/mock_spec/fixed/seed0")


def test_generate_resolutions_simple() -> None:
    """
    When there are no "@" variables in the config dict, we get just the config dict back, with
    an empty resolutions string.
    """
    config_dct = {"foo": "one", "bar": "two"}
    result = list(settings.generate_resolutions(config_dct, None, None))
    assert result == [("", config_dct)]


def test_generate_resolutions_both() -> None:
    """
    When there is a single "@" variable and no restrictions, we get both values back.
    """
    config_dct = {"foo": ["@x", "one", "other"], "bar": "two", "results_dir": Path("res")}
    result = list(settings.generate_resolutions(config_dct, None, None))
    assert result == [
        ("x1", {"foo": "one", "bar": "two", "results_dir": Path("res") / "x1", "resolution_spec": "x1"}),
        ("x2", {"foo": "other", "bar": "two", "results_dir": Path("res") / "x2", "resolution_spec": "x2"}),
    ]


def test_generate_resolutions_x2() -> None:
    """
    When we specify one of the resolutions, we get just that resolution back.
    """
    config_dct = {"foo": ["@x", "one", "other"], "bar": "two"}
    result = list(settings.generate_resolutions(config_dct, None, "x2"))
    assert result == [("x2", {"foo": "other", "bar": "two", "resolution_spec": "x2"})]


def test_generate_resolutions_max1() -> None:
    """
    When we specify to get at most one resolution back, we get one of the two possible results.
    """
    config_dct = {"foo": ["@x", "one", "other"], "bar": "two"}
    result = list(settings.generate_resolutions(config_dct, 1, None))
    assert result in [
        [("x1", {"foo": "one", "bar": "two", "resolution_spec": "x1"})],
        [("x2", {"foo": "other", "bar": "two", "resolution_spec": "x2"})],
    ]


def test_files_repeat() -> None:
    """If the same CSV file repeats, raise a ValidationError"""
    files = {"1": "somefile.csv", "2": "somefile.csv"}

    with pytest.raises(pydantic.ValidationError):
        abex.data_settings.DataSettings(files=files)
