# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import staticchar as ch
from psbutils.misc import find_subrepo_directory

SUBREPO_DIR = find_subrepo_directory()


def test_load_config() -> None:
    """Assert that given a config name, it is located and parsed correctly"""
    spec_name = "integral_basic.yml"
    spec_filename = SUBREPO_DIR / "tests" / "configs" / spec_name
    config = ch.config.load(spec_filename, ch.config.CharacterizationConfig)
    assert config.method == ch.config.CharacterizationMethod.Integral
    assert config.growth_signal == "OD"
    assert config.reference == "mRFP1"
    assert config.signal_properties["EYFP"].color == "gold"
    assert set(config.background_subtract_columns()) == set(["OD", "EYFP", "ECFP", "mRFP1"])
    assert config.colors() == {"OD": "black", "EYFP": "gold", "ECFP": "cyan", "mRFP1": "red"}
