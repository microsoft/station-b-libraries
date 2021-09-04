# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from pathlib import Path

import abex.scripts
import numpy as np
import pandas as pd
import pytest
from abex.optimizers.zoom_optimizer import interval_length, shift_to_within_parameter_bounds, shrink_interval
from psbutils.misc import find_subrepo_directory
from slow_tests.test_abex_tutorials import write_tutorial_data


def test_shrink_interval():
    initial_interval = (0, 100)
    optimum = 60
    shrinking_factor = 0.1

    new_interval = shrink_interval(
        shrinking_factor=shrinking_factor, interval=initial_interval, shrinking_anchor=optimum
    )

    assert pytest.approx(optimum) == (new_interval[0] + new_interval[1]) / 2

    expected_length = shrinking_factor * interval_length(initial_interval)
    assert new_interval[1] - new_interval[0] == pytest.approx(expected_length)


def test_shrink_and_shift_interval():
    initial_interval = (0, 100)
    optimum = 90
    shrinking_factor = 0.6

    shrunk_interval = shrink_interval(
        shrinking_factor=shrinking_factor, interval=initial_interval, shrinking_anchor=optimum
    )

    new_interval = shift_to_within_parameter_bounds(new_interval=shrunk_interval, old_interval=initial_interval)

    assert interval_length(shrunk_interval) == pytest.approx(interval_length(new_interval))
    assert new_interval[0] >= initial_interval[0] and new_interval[1] <= initial_interval[1]


# TODO: It may be beneficial to create an artificial data set, to see whether the whole ``run`` works as
#  expected. (Right now it is tested through function optimization using data loop, but we may check more things).
def generate_tutorial_zoomopt_data(seed, x_max, basename):
    np.random.seed(seed)
    batch_size = 5
    x = np.linspace(0, x_max, batch_size)
    eps = np.random.rand(batch_size)
    objective = 2 - (x - 0.42) ** 2 + eps
    collected_data = pd.DataFrame({"x": x})
    collected_data["objective"] = objective
    write_tutorial_data(collected_data, basename)


def check_zoomopt_results(x_max: float) -> None:
    res_path = Path("Results") / "tutorial-zoomopt"
    for seed_dir in res_path.glob("fixed/seed*"):
        batch_path = seed_dir / "batch.csv"
        with batch_path.open() as f:
            lines = [line.rstrip() for line in f.readlines()]
            assert lines[0] == "x"
            assert sorted(float(v) for v in lines[1:]) == [0.25 * i * x_max for i in range(5)]
        optima_path = seed_dir / "optima.csv"
        with optima_path.open() as f:
            lines = [line.rstrip() for line in f.readlines()]
            assert lines == ["x", "0.0"]


@pytest.mark.timeout(10)
def test_tutorial_zoomopt():
    generate_tutorial_zoomopt_data(24601, 10, "tutorial-zoomopt.csv")
    subrepo_dir = find_subrepo_directory()
    abex.scripts.run.main(["--spec_file", f"{subrepo_dir}/tests/data/Specs/tutorial-zoomopt.yml"])
    check_zoomopt_results(5.0)
    generate_tutorial_zoomopt_data(42, 5, "tutorial-zoomopt-1.csv")
    abex.scripts.run.main(["--spec_file", f"{subrepo_dir}/tests/data/Specs/tutorial-zoomopt-1.yml"])
    check_zoomopt_results(2.5)
