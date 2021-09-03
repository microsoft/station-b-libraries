# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from typing import Tuple

import numpy as np
from numpy.testing import assert_array_equal
import matplotlib.pyplot as plt

plt.switch_backend("Agg")
from abex.plotting import core  # noqa: E402
from psbutils.filecheck import figure_found  # noqa: E402


def summer(arr: np.ndarray):
    return arr.sum(axis=1)


def test_calc_1d_slice():
    # We'll slice through the second dimension (dim=1 below) so the 3 is irrelevant.
    slice_loc = np.array([2, 3, 4]).astype(np.double)
    # Calculate at (2,2.0,4), (2,2.5,4), (2,3.0,4), (2,3.5,4), (2,4.0,4)
    low = 2.0
    high = 4.0
    resolution = 5
    slice1d_x, slice1d_f = core.calc_1d_slice(
        summer, dim=1, slice_loc=slice_loc, slice_bounds=(low, high), resolution=resolution
    )
    # x values are five points in range (2, 4)
    assert_array_equal(slice1d_x, np.linspace(low, high, resolution))
    # f values are 2+x+4 for x as above, i.e. x+6
    assert_array_equal(slice1d_f, slice1d_x + 6.0)
    ax = core.plot_1d_slice_through_function(
        summer, dim=1, slice_loc=slice_loc, slice_bounds=(low, high), resolution=resolution
    )
    assert figure_found(ax, "plot_1d_slice_through_function")


def test_calc_2d_slice():
    # We'll slice through the second and third dimensions (x_dim=1, y_dim=2 below) so the 3 and 4 are irrelevant.
    slice_loc = np.array([2, 3, 4, 5]).astype(np.double)
    # Calculate at (2,x,y,5) for x and y in the ranges indicated (25 po
    x_low = 2.0
    x_high = 4.0
    y_low = 3.0
    y_high = 5.0
    resolution = 5
    slice2d_x, slice2d_y, slice2d_f = core.calc_2d_slice(
        summer,
        dim_x=1,
        dim_y=2,
        slice_loc=slice_loc,
        slice_bounds_x=(x_low, x_high),
        slice_bounds_y=(y_low, y_high),
        resolution=resolution,
    )
    # x values are five points in range (2, 4)
    assert slice2d_x.shape == (5, 5)
    for i in range(5):
        assert_array_equal(slice2d_x[i, :], np.linspace(x_low, x_high, resolution))
        assert_array_equal(slice2d_y[:, i], np.linspace(y_low, y_high, resolution))
    # y values are 2+x+y+5 for x as above, i.e. x+y+7
    assert_array_equal(slice2d_f, slice2d_x + slice2d_y + 7)
    ax, _ = core.plot_2d_slice_through_function(
        summer,
        dim_x=1,
        dim_y=2,
        slice_loc=slice_loc,
        slice_bounds_x=(x_low, x_high),
        slice_bounds_y=(y_low, y_high),
        resolution=resolution,
    )
    assert figure_found(ax, "plot_2d_slice_through_function")


def test_plot_1d_slice_through_function_with_confidence_intervals():
    def func(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = summer(arr)
        return mean, mean - 1, mean + 1

    # We'll slice through the second dimension (dim=1 below) so the 3 is irrelevant.
    slice_loc = np.array([2, 3, 4]).astype(np.double)
    # Calculate at (2,2.0,4), (2,2.5,4), (2,3.0,4), (2,3.5,4), (2,4.0,4)
    low = 2.0
    high = 4.0
    resolution = 5
    ax = core.plot_1d_slice_through_function_with_confidence_intervals(
        func, dim=1, slice_loc=slice_loc, slice_bounds=(low, high), resolution=resolution
    )
    assert figure_found(ax, "plot_1d_slice_through_function_with_confidence_intervals")


def test_plot_2d_slice_from_arrays():
    x, y, z = core.calc_2d_slice(
        lambda a: np.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2),
        dim_x=0,
        dim_y=1,
        slice_loc=np.array([0, 0]),
        slice_bounds_x=(-1, 1),
        slice_bounds_y=(-1, 1),
        resolution=5,
    )
    ax, _ = core.plot_2d_slice_from_arrays(x, y, z)
    assert figure_found(ax, "plot_2d_slice_from_arrays")
