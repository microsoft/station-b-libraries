# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import random
from typing import Tuple
import numpy as np
import matplotlib
import pytest

from abex.expand import FixedSeed

matplotlib.use("Agg")

from abex.plotting import composite_core  # noqa: E402
from psbutils.filecheck import figure_found  # noqa: E402


def cos_of_sum(arr: np.ndarray) -> np.ndarray:
    return np.cos(arr.sum(axis=1)).reshape(-1, 1)


def get_cos_of_sum_data() -> Tuple[np.ndarray, np.ndarray]:
    # We use random.gauss and convert it to numpy, rather than using np.random.randn, because
    # we need a fixed seed, we have the FixedSeed context manager, and np.random.set_state is a pain to use.
    with FixedSeed(0):
        values = [random.gauss(0.0, 1.0) for _ in range(90)]
        x = np.clip(np.array(values).reshape(30, 3), -1, 1)
    assert isinstance(x, np.ndarray)  # for mypy
    y = cos_of_sum(x)
    return x, y


def cos_of_sum_with_bounds(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = cos_of_sum(arr)
    return mean, mean - 0.1, mean + 0.1


@pytest.mark.timeout(10)
def test_plot_multidimensional_function_slices():
    fig, _ = composite_core.plot_multidimensional_function_slices(
        cos_of_sum, slice_loc=np.zeros((3,)), bounds=[(-1.0, 1.0)] * 3
    )
    assert figure_found(fig, "plot_multidimensional_function_slices")


@pytest.mark.timeout(10)
def test_plot_multidimensional_function_slices_confidence_bounds():
    fig, _ = composite_core.plot_multidimensional_function_slices(
        cos_of_sum_with_bounds,
        slice_loc=np.zeros((3,)),
        bounds=[(-1.0, 1.0)] * 3,
        func_returns_confidence_intervals=True,
    )
    assert figure_found(fig, "plot_multidimensional_function_slices_confidence_bounds")


@pytest.mark.timeout(10)
def test_plot_slices1d_with_binned_data():
    x, y = get_cos_of_sum_data()
    fig = composite_core.plot_slices1d_with_binned_data(
        x,
        y,
        dim_x=0,
        slice_dim1=1,
        slice_dim2=2,
        slice_loc=np.zeros((3,)),
        bounds=[(-1.0, 1.0)] * 3,
        num_slices=3,
        predict_func=cos_of_sum_with_bounds,
    )
    assert figure_found(fig, "plot_slices1d_with_binned_data")


@pytest.mark.timeout(10)
def test_plot_projected_slices1d():
    fig, _ = composite_core.plot_projected_slices1d(
        cos_of_sum,
        slice_loc=3.0 * np.ones((3,)),
        bounds=[(2.0, 4.0)] * 3,
        dim_x=0,
        slice_dim=1,
        projection_dim=2,
        input_names=["x1", "x2", "x3"],
    )
    assert figure_found(fig, "plot_projected_slices1d")
