# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import numpy as np
import staticchar as ch
from psbutils.misc import find_subrepo_directory


SUBREPO_DIR = find_subrepo_directory()


def test_integral_well_basic() -> None:
    """Assert that the integral method produces known values"""
    # Load data
    data_name = "S-shape"
    data_folder = SUBREPO_DIR / "tests" / "test_data" / data_name
    data = ch.Dataset(data_folder)

    # Load config
    spec_name = "integral_basic.yml"
    spec_filename = SUBREPO_DIR / "tests" / "configs" / spec_name
    config = ch.config.load(spec_filename, ch.config.CharacterizationConfig)

    # Integral method for a single frame
    example = data.get_a_frame()
    subtracted = ch.subtract_background(
        example, columns=config.background_subtract_columns(), strategy=ch.BackgroundChoices.Minimum
    )
    integral = ch.integrate(data=subtracted, signals=config.signals, interval=config.time_window)
    rounded_integral = {k: np.round(v) for k, v in integral.items()}
    expected_integral = {"EYFP": 4096, "ECFP": 8415}
    assert rounded_integral == expected_integral
