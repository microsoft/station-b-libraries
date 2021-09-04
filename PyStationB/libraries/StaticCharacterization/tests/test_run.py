# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import shutil

import pytest
from psbutils.filecheck import figure_found
from staticchar.config import CharacterizationConfig, load
from staticchar.plate import (
    CHARACTERIZATIONS_CSV,
    RANK_CORRELATIONS,
    GROWTH_MODEL,
    INTEGRATION_FMT,
    MODEL_VALUES_CSV,
    SIGNALS_VS_REFERENCE,
    SIGNALS_VS_TIME,
    VALUE_CORRELATIONS,
)
from staticchar.run import CONFIG_YML, OUTPUT_DIRECTORY, run
from psbutils.misc import find_subrepo_directory


@pytest.mark.timeout(40)
def test_run():
    subrepo_dir = find_subrepo_directory()
    integral_path = subrepo_dir / "tests/configs/integral_basic.yml"
    paths = [
        subrepo_dir / "tests/configs/gradient.yml",
        integral_path,
        subrepo_dir / "tests/test_data/S-shape/",
    ]
    output_dirs = run(paths)
    assert len(output_dirs) == 2
    gradient_dir, integral_dir = output_dirs
    assert gradient_dir.parent == OUTPUT_DIRECTORY / "gradient" / "S-shape"
    assert integral_dir.parent == OUTPUT_DIRECTORY / "integral_basic" / "S-shape"
    gradient_bases = set(p.name for p in gradient_dir.glob("*"))
    common_set = set(
        [
            CHARACTERIZATIONS_CSV,
            CONFIG_YML,
            MODEL_VALUES_CSV,
            SIGNALS_VS_TIME + ".png",
            GROWTH_MODEL + ".png",
            RANK_CORRELATIONS + ".png",
            VALUE_CORRELATIONS + ".png",
        ]
    )
    assert gradient_bases == common_set.union([SIGNALS_VS_REFERENCE + ".png"])
    integral_bases = set(p.name for p in integral_dir.glob("*"))
    config = load(integral_path, CharacterizationConfig)
    assert integral_bases == common_set.union(INTEGRATION_FMT.format(signal) + ".png" for signal in config.signals)
    all_found = True
    for output_dir in output_dirs:
        cfg_type = output_dir.parent.parent.name  # gradient or integral_basic
        for png_file in output_dir.glob("*.png"):
            name = png_file.name[: -len(".png")]
            if not figure_found(png_file, f"test_run_{cfg_type}_{name}"):
                all_found = False
    if all_found:
        shutil.rmtree(gradient_dir.parent)
        shutil.rmtree(integral_dir.parent)
    assert all_found
