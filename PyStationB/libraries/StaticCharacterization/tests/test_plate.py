# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import staticchar as ch
from psbutils.misc import find_subrepo_directory


SUBREPO_DIR = find_subrepo_directory()


@pytest.mark.timeout(30)
def test_gradient_plate() -> None:
    """Test running the gradient method on a whole plate"""
    # Load data
    data_name = "S-shape"
    data_folder = SUBREPO_DIR / "tests" / "test_data" / data_name
    data = ch.Dataset(data_folder)

    # Load config
    spec_name = "gradient.yml"
    spec_filename = SUBREPO_DIR / "tests" / "configs" / spec_name
    config = ch.config.load(spec_filename, ch.config.CharacterizationConfig)

    plate = ch.Plate(data, config)
    with TemporaryDirectory() as tmpd:
        plate.characterize(Path(tmpd))
    assert len(plate) == 3


@pytest.mark.timeout(30)
def test_integral_plate() -> None:
    """Test running the integral method on a whole plate"""
    # Load data
    data_name = "S-shape"
    data_folder = SUBREPO_DIR / "tests" / "test_data" / data_name
    data = ch.Dataset(data_folder)

    # Load config
    spec_name = "integral_basic.yml"
    spec_filename = SUBREPO_DIR / "tests" / "configs" / spec_name
    config = ch.config.load(spec_filename, ch.config.CharacterizationConfig)

    plate = ch.Plate(data, config)
    with TemporaryDirectory() as tmpd:
        plate.characterize(Path(tmpd))
    assert len(plate) == 3


@pytest.mark.timeout(30)
def test_integral_plate_corrected() -> None:
    """Test running the integral method on a whole plate"""
    # Load data
    data_name = "S-shape"
    data_folder = SUBREPO_DIR / "tests" / "test_data" / data_name
    data = ch.Dataset(data_folder)

    # Load config
    spec_name = "integral_time_corrected.yml"
    spec_filename = SUBREPO_DIR / "tests" / "configs" / spec_name
    config = ch.config.load(spec_filename, ch.config.CharacterizationConfig)

    assert len(config.reference_wells) == 1

    plate = ch.Plate(data, config)
    with TemporaryDirectory() as tmpd:
        plate.characterize(Path(tmpd))
    assert len(plate) == 3

    # Test accessing a result by ID
    res = plate[config.reference_wells[0]]
    assert np.round(res["EYFP"]) == 4931
    assert np.round(res["ECFP"]) == 10556

    # Test iterating through results, and check that source data exists (and arbitrarily has several rows)
    for k, v in plate.items():
        len(data[k][ch.TIME]) > 10  # type: ignore # auto

    # Test creating a dataframe
    df = plate.to_dataframe()
    assert len(df) == 3
