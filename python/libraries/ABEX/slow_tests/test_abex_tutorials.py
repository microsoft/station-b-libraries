# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import os
from pathlib import Path

import abex.scripts.run
import numpy as np
import pandas as pd
import pytest
from abex.plotting.expected_basenames import expected_basenames_1d, expected_basenames_2d
from psbutils.misc import find_subrepo_directory

SUBREPO_DIR = find_subrepo_directory()


def generate_tutorial_intro_data():
    np.random.seed(42)
    n_observations = 50
    x = np.random.rand(n_observations)
    y = np.random.rand(n_observations)
    eps = np.random.normal(scale=0.05, size=n_observations)
    objective = 4 - (x - 0.2) ** 2 - (y - 0.6) ** 2 + eps
    collected_data = pd.DataFrame({"x": x, "y": y})
    collected_data["objective"] = objective
    write_tutorial_data(collected_data, "tutorial-intro-data.csv")


@pytest.mark.timeout(2400)
@pytest.mark.skip("Can cause ADO timeout on both Windows and Linux")
def test_tutorial_intro():
    generate_tutorial_intro_data()
    # Equivalent to: python scripts/run.py --spec_file tutorial-intro.yml
    abex.scripts.run.main(["--spec_file", f"{SUBREPO_DIR}/tests/data/Specs/tutorial-intro.yml"])
    found = [os.path.basename(p) for p in sorted((Path("Results") / "tutorial-intro").glob("*.???"))]
    assert found == expected_basenames_2d(3, variant=1)


def generate_tutorial_intro_data_advanced():
    np.random.seed(42)
    n_observations = 50
    logx = 4 * np.random.rand(n_observations)
    eps = np.random.normal(scale=0.2, size=n_observations)
    objective = 1200 * np.exp(np.cos(np.pi * (logx - 1)) + eps)
    collected_data = pd.DataFrame({"x": 10 ** logx})
    collected_data["objective"] = objective
    write_tutorial_data(collected_data, "tutorial-intro-2.csv")


def write_tutorial_data(data: pd.DataFrame, basename: str) -> None:
    data_dir = Path("Tutorial") / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_dir / basename, index=False)


# @pytest.mark.timeout(1500)
@pytest.mark.skip("Can cause ADO timeout on both Windows and Linux")
def test_tutorial_intro_advanced() -> None:
    generate_tutorial_intro_data_advanced()
    abex.scripts.run.main(["--spec_file", f"{SUBREPO_DIR}/tests/data/Specs/tutorial-intro-2.yml"])
    found = [os.path.basename(p) for p in sorted((Path("Results") / "tutorial-intro-2").glob("*.???"))]
    assert found == expected_basenames_1d(3, variant=1)
