# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
from abex.plotting.convergence_plotting import (
    get_color_groups,
    get_color_palette,
    get_max_per_batch,
    get_run_names_palette,
)
from abex.simulations.plot_convergence_multiple_runs import SEED_COLUMN


@pytest.fixture
def df():
    experiment_names = ["run1", "run2", "run3", "run4", "run5"]
    num_experiments = len(experiment_names)
    return pd.DataFrame(
        {
            "Experiment Name": experiment_names,
            "Batch Number": [1] * num_experiments,
            "Objective": np.random.rand(num_experiments),
            "batch_strategy": np.random.choice(["LocalPenalization", "MomentMatchedEI"], size=num_experiments),
            "acquisition": np.random.choice(
                ["EXPECTED_IMPROVEMENT", "MEAN_PLUGIN_EXPECTED_IMPROVEMENT"], size=num_experiments
            ),
            "optimization_strategy": np.random.choice(["Bayes", "Zoom"], size=num_experiments),
        }
    )


def test_get_color_groups(df):
    style_col = "batch_strategy"
    run_col = "Experiment Name"
    color_groups = get_color_groups(df, [style_col], run_col)
    assert len(color_groups) == len(df[style_col].unique())
    assert len([x for y in color_groups for x in y]) == len(df[run_col])

    style_cols = ["batch_strategy", "optimization_strategy"]
    color_groups = get_color_groups(df, style_cols, run_col)
    assert len(color_groups) > 0
    # ensure what's returned is a list of lists of strings
    assert isinstance(color_groups, list)
    assert isinstance(color_groups[0], list)
    assert isinstance(color_groups[0][0], str)
    assert len([x for y in color_groups for x in y]) == len(df[run_col])


def test_get_color_palette() -> None:
    run_names = ["x", "y", "z"]
    palette = get_color_palette(run_names)
    assert len(palette) == len(run_names)
    assert isinstance(palette[0], tuple)
    assert isinstance(palette[0][0], float)

    run_groups = [["x", "y"], ["z"]]
    palette = get_color_palette(run_names)
    # still expect 1 colour per run_name (e.g. x, y, z)
    assert len(palette) == len([x for y in run_groups for x in y])
    assert isinstance(palette[0], tuple)
    assert isinstance(palette[0][0], float)

    palette = get_color_palette([])  # type: ignore
    assert len(palette) == 0


def test_get_run_names_palette(df, monkeypatch):
    style_col = []
    run_col = "Experiment Name"
    run_names, palette = get_run_names_palette(df, style_col, run_col)
    assert isinstance(run_names, list)
    assert isinstance(run_names[0], str)
    assert sorted(df[run_col].unique()) == sorted(run_names)
    assert len(palette) == len(df[run_col].unique())
    assert isinstance(palette, list)
    assert isinstance(palette[0], tuple)
    assert isinstance(palette[0][0], float)

    style_col = ["batch_strategy"]
    run_names, palette = get_run_names_palette(df, style_col, run_col)
    # the run_names and palette will be the same length (1 color per run name)
    assert isinstance(run_names, list)
    assert isinstance(run_names[0], str)
    assert sorted(df[run_col].unique()) == sorted(run_names)
    assert len(palette) == len(df[run_col].unique())
    assert isinstance(palette, list)
    assert isinstance(palette[0], tuple)
    assert isinstance(palette[0][0], float)


def test_get_max_per_batch(df):
    run_col = "Experiment Name"
    seed_col = SEED_COLUMN
    batch_num_col = "Batch Number"

    # generate 2 subruns per run
    num_rows = len(df[run_col])
    seeds = [np.random.randint(10) for _ in range(num_rows // 2)] * 2
    if num_rows % 2 == 1:
        seeds += [seeds[0]]
    df[seed_col] = seeds
    run_names = df[run_col].unique().tolist()

    max_per_batch_df = get_max_per_batch(df, run_names, run_col, seed_col, batch_num_col)
    assert len(max_per_batch_df) > 0
