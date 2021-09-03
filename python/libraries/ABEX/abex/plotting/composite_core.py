# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
A file for plots that combine the functionality in core.py to create composite (possibly multi-panelled - with
multiple subplots) plots visualising a multi-dimensional function or predictor in various ways.

This file still attempts to remain as application agnostic as possible, without references to Bayes. Opt.
or any applications in general. It can in principle be used to visualise any function or predictor from a
multidimensional input space to a single output. Or any data with multiple inputs + 1 output to visualise.
"""
import functools
from typing import Callable, List, Optional, Tuple, Union

import matplotlib
import matplotlib.colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from abex.plotting.core import (
    PLOT_SCALE,
    NDAorTuple,
    calc_1d_slice,
    calc_2d_slice,
    make_lower_triangular_axis_grid_with_colorbar_axes,
    plot_1d_slice_through_function,
    plot_1d_slice_through_function_with_confidence_intervals,
    plot_2d_slice_from_arrays,
)


def plot_multidimensional_function_slices(
    func: Callable[[np.ndarray], NDAorTuple],
    slice_loc: np.ndarray,
    bounds: Union[np.ndarray, List[Tuple[float, float]]],
    input_names: Optional[List[str]] = None,
    obs_points: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    input_scales: Optional[List[PLOT_SCALE]] = None,
    output_scale: PLOT_SCALE = "linear",
    output_label: str = "Objective Value",
    size: float = 3,
    slice_2d_resolution: int = 50,
    # slide_1d_resolution: int = 100,
    func_returns_confidence_intervals: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots a multidimensional function across each slice aligned with principal axes through the slice_point (usually the
    optimum of objective). Also scatter-plots previously observed points over the slices (if those are supplied).

    Example:

    .. image:: ../../tests/data/figures/plot_multidimensional_function_slices_confidence_bounds/figure000.png

    Args:
        func: The multidimensional function to plot slices of. Returns either a single array of shape
            [num_inputs, 1], or if it's a predictor, returns 3 arrays of shape [num_inputs, 1] corresponding
            to mean, lower-bounds and upper-bound predictions respectively. If the function does return
            lower and upper-bound predictions, `func_returns_confidence_intervals` must be set to True.
        slice_loc: The input coordinates at which to take the slices
        bounds: Array (of shape [num_inputs, 2]) or list of bounds for each input dimension.
            E.g. for two inputs x1 within [0.1, 1.0], x2 within [-1. 2] the bounds would be: [(0.1, 1.), (-1, 2)]
        input_names (optional): Names of the input parameters (used for labels). Defaults to None.
        obs_points (optional): Points observed during optimization procedure. If given as a list, each element
            is assumed to correspond to one batch (the points will be coloured by batch). These points will be scattered
            over the slices. Defaults to None.
        input_scales (optional): Scales of each input dimension (log, symlog, linear...). Defaults to linear for all.
        output_scale (optional): Scale of the output dimension. Defaults to "linear".
        output_label (optional): The name for the output value to use for the colourbar.
        size (optional): Size of each slice subplot. Defaults to 3.
        slice_2d_resolution (optional): Grid resolution for the 2d plots.
        # slice_1d_resolution (optional): Grid resolution for the 1d slice plots.
        func_returns_confidence_intervals (optional): whether func returns a tuple of ndarrays rather than an ndarray.

    Returns:
        Figure object with the plot and an array of axes with each slice.
    """
    # Input validation checks
    assert output_scale in ["linear", "log", "symlog"]

    def func_return_just_mean(x):
        """
        If the supplied function is a predictor returning lower and upper confidence bounds as well as mean,
        return just the mean prediction. If not, return the function value evaluated at x.
        """
        return func(x)[0] if func_returns_confidence_intervals else func(x)

    n_dims: int = len(bounds)
    # If multiple batches of points supplied as a list in obs_points, make a colour palette
    n_batches = len(obs_points) if isinstance(obs_points, (list, tuple)) else 1
    scatter_colours = sns.color_palette("viridis", n_colors=n_batches)
    # If input_scales not specified, default all to 'linear'
    input_scales = input_scales if input_scales else ["linear"] * n_dims  # type: ignore # auto
    # Keep track of contour sets returned for each axis
    contour_sets = []

    # Construct axes
    fig = plt.figure(figsize=(size * n_dims, size * n_dims))
    axes, cbar_axes = make_lower_triangular_axis_grid_with_colorbar_axes(
        fig=fig, num_cols=n_dims, num_colorbars=2, share_y_on_diagonal=True
    )

    # Keep a running minimum and maximum of function values in 2D slices
    func_values_min: float = np.inf
    func_values_max: float = -np.inf

    with sns.axes_style("darkgrid"):
        for i in range(n_dims):  # i iterates over the rows of the plots
            for j in range(n_dims):  # j iterates over the columns of the plots
                ax = axes[i, j]
                # 1D-slice plots along the diagonal
                if i == j:
                    if func_returns_confidence_intervals:
                        plot_1d_slice_through_function_with_confidence_intervals(
                            func,  # type: ignore
                            dim=i,
                            slice_loc=slice_loc,
                            slice_bounds=bounds[i],
                            ax=ax,
                            x_scale=input_scales[i],
                        )
                    else:
                        plot_1d_slice_through_function(
                            func,  # type: ignore
                            dim=i,
                            slice_loc=slice_loc,
                            slice_bounds=bounds[i],
                            ax=ax,
                            x_scale=input_scales[i],
                        )
                    ax.set_yscale(output_scale)

                # lower triangle
                elif i > j:
                    dim_x, dim_y = j, i
                    # Compute the data for the 2D slice plots
                    xx, yy, func_values_slice = calc_2d_slice(
                        func=func_return_just_mean,  # type: ignore # auto
                        dim_x=dim_x,
                        dim_y=dim_y,
                        slice_loc=slice_loc,
                        slice_bounds_x=bounds[dim_x],
                        slice_bounds_y=bounds[dim_y],
                        x_scale=input_scales[dim_x],
                        y_scale=input_scales[dim_y],
                        resolution=slice_2d_resolution,
                    )
                    # Plot the 2D slice
                    _, im = plot_2d_slice_from_arrays(
                        xx,
                        yy,
                        func_values_slice,
                        ax=ax,
                        x_scale=input_scales[dim_x],
                        y_scale=input_scales[dim_y],
                        output_scale=output_scale,
                    )
                    contour_sets.append(im)
                    # Keep a running minimum and maximum of function values in slices
                    func_values_min = min(func_values_min, func_values_slice.min())  # type: ignore
                    func_values_max = max(func_values_max, func_values_slice.max())  # type: ignore
                    # Scatter points on the slices if given
                    if obs_points is not None:  # pragma: no cover
                        if isinstance(obs_points, np.ndarray):
                            # If just one array given, scatter with the colour reflecting objective value
                            ax.scatter(
                                obs_points[:, dim_x], obs_points[:, dim_y], color=scatter_colours[0], s=20, zorder=15
                            )
                        else:
                            assert isinstance(obs_points, (list, tuple))
                            # If multiple arrays given, colour the points according to the batch number
                            for batch_num, batch_arr in enumerate(obs_points):
                                ax.scatter(
                                    batch_arr[:, dim_x],
                                    batch_arr[:, dim_y],
                                    color=scatter_colours[batch_num],
                                    s=25,
                                    lw=0.0,
                                    alpha=0.8,
                                    zorder=15,
                                )
                # Add axis labels
                if input_names is not None:  # pragma: no cover
                    # If plot in the first column (but not first row), add a y_label
                    if i != 0 and j == 0:
                        axes[i, j].set_ylabel(input_names[i])
                    # If plot is at the bottom, add an x_label
                    if i == n_dims - 1:
                        axes[i, j].set_xlabel(input_names[j])
                if i >= j:
                    # Remove redundant ticks on inner plots
                    if i != n_dims - 1:
                        axes[i, j].xaxis.set_visible(False)
                    if j != 0:
                        axes[i, j].yaxis.set_visible(False)
                    # # Prune the upper-most tick from plot, so that the ticks don't overlap each other between plots
                    # ax.yaxis.set_major_locator(ticker.MaxNLocator(prune='upper'))
                    ax.tick_params(axis="both", which="major", labelsize=9)
                    ax.tick_params(axis="both", which="minor", labelsize=6)
        # Update the colour limits of the slice plots
        for contour_set in contour_sets:
            contour_set.set_clim(vmin=func_values_min, vmax=func_values_max)
        # Add the colourbars
        if n_dims > 1:
            # make a colourbar for the contour plots
            cb1 = fig.colorbar(contour_sets[-1], cax=cbar_axes[0], aspect=50)
            cb1.set_label(output_label)
            cbar_axes[0].yaxis.set_ticks_position("left")
            # make a colourbar for different batches
            if n_batches > 1:  # pragma: no cover
                cb2 = matplotlib.colorbar.ColorbarBase(  # type: ignore # auto
                    cbar_axes[1],
                    cmap=matplotlib.colors.ListedColormap(scatter_colours),
                    boundaries=[x - 0.5 for x in range(n_batches + 1)],
                    ticks=list(range(n_batches)),
                    spacing="proportional",
                )
                cb2.set_label("Batch Number")
            else:
                cbar_axes[1].set_visible(False)
    return fig, axes


def plot_projected_slices1d(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: Union[np.ndarray, List[Tuple[float, float]]],
    slice_loc: np.ndarray,
    dim_x: int,
    slice_dim: int,
    projection_dim: int,
    input_names: Optional[List[str]] = None,
    num_slices: int = 6,
    num_projections: int = 6,
    size: float = 4,
    scale: PLOT_SCALE = "log",
    input_scales: Optional[List[PLOT_SCALE]] = None,
    output_scale: PLOT_SCALE = "linear",
    output_label: str = "Objective",
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Visualises a multidimensional function with a series of slice1d plots. It plots num_slices subplots,
    each sub-plot plotting the output on the y-axis against input dimensions dim_x on the x-axis. Multiple
    curves will be plotted on each sub-plot, corresponding to slices at different values of the input referenced
    by projection_dim. Each sub-plot correspond to slices at different values of the input referenced by slice_dim.

    Example:

    .. image:: ../../tests/data/figures/plot_projected_slices1d/figure000.png

    Args:
        func: The multidimensional function to plot slices of
        bounds: Array (of shape [num_inputs, 2]) or list of bounds for each input dimension.
            E.g. for two inputs x1 within [0.1, 1.0], x2 within [-1. 2] the bounds would be: [(0.1, 1.), (-1, 2)]
        slice_loc: The input coordinates at which to take the slices
        dim_x: The index of input to put on the x-axis.
        slice_dim: The index of the input to vary across sub-plots.
        projection_dim: The index of the input to vary within each sub-plot - plotting a different curve for each value.
        input_names (optional): Names of the input parameters (used for labels). Defaults to None.
        num_slices (int, optional): Num subplots, each corresponding to the input at index slice_dim being
            fixed to one value. Defaults to 6.
        num_projections (int, optional): The number of curves corresponding to different slices in each
            sub-plot. Defaults to 6.
        size (optional): Size of each slice subplot.
        input_scales (optional): Scales of each input dimension (log, symlog, linear...). Defaults to linear for all.
        output_scale (optional): Scale of the output dimension. Defaults to "linear".
        output_label (optional): The name for the output value to use for the colourbar.

    Returns:
        Figure object with the plot.
    """
    # Input validation checks
    assert output_scale in ["linear", "log", "symlog"]
    assert len({slice_dim, projection_dim, dim_x}) == 3, "slice_dim, project_dim and dim_x must be different."

    n_dims: int = len(bounds)
    input_scales = input_scales if input_scales else ["linear"] * n_dims  # type: ignore # auto

    # Construct axes
    fig = plt.figure(figsize=(size * num_slices, size))
    spec = gridspec.GridSpec(
        ncols=num_slices + 1, nrows=1, figure=fig, width_ratios=[1.1] * (num_slices) + [0.15]  # type: ignore
    )
    spec.update(wspace=0.1, hspace=0.15)
    axes = np.empty([num_slices], dtype=object)
    slice_dim_grid = (
        np.geomspace(bounds[slice_dim][0], bounds[slice_dim][1], num_slices + 2)
        if scale == "log"
        else np.linspace(bounds[slice_dim][0], bounds[slice_dim][1], num_slices + 2)
    )[1:-1]
    proj_dim_grid = (
        np.geomspace(bounds[projection_dim][0], bounds[projection_dim][1], num_projections)
        if scale == "log"
        else np.linspace(bounds[projection_dim][0], bounds[projection_dim][1], num_projections)
    )

    norm = matplotlib.colors.LogNorm(
        vmin=bounds[projection_dim][0], vmax=bounds[projection_dim][1]  # type: ignore # auto
    )  # type: ignore # auto
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=sns.color_palette("flare", as_cmap=True))  # type: ignore # auto
    ax = axes[0]  # to avoid "accessed before assignment" warnings
    for i in range(num_slices):  # Iterates over the columns of the plot
        if i == 0:
            axes[i] = fig.add_subplot(spec[i])
        else:
            axes[i] = fig.add_subplot(spec[i], sharey=axes[i - 1], sharex=axes[i - 1])
        ax = axes[i]
        for j in range(num_projections):
            curve_slice_loc = slice_loc.copy()
            curve_slice_loc[slice_dim] = slice_dim_grid[i]
            curve_slice_loc[projection_dim] = proj_dim_grid[j]

            x, func_values = calc_1d_slice(
                func=func,
                dim=dim_x,
                slice_loc=curve_slice_loc,
                slice_bounds=bounds[dim_x],
                x_scale=scale,
            )
            ax.plot(x, func_values, color=cmap.to_rgba(proj_dim_grid[j]), linewidth=4, alpha=0.7)

        # Add axis labels
        if input_names is not None:
            # If plot in the first column (but not first row), add a y_label
            if i == 0:
                ax.set_ylabel(output_label)
            ax.set_xlabel(input_names[dim_x])
        # Remove redundant ticks
        if i > 0:
            ax.tick_params(labelleft=False)
        ax.set_title((input_names[slice_dim] if input_names else "") + f" = {slice_dim_grid[i]:3f}")
        # Update norm limits for colour scaling for each axis:
        # Set scales (logarithmic, symlog, linear etc.)
        ax.set_xscale(input_scales[dim_x])
    ax.set_yscale(output_scale)
    # Limits
    ax.set_xlim(*bounds[dim_x])

    ax_cbar = fig.add_subplot(spec[-1])
    # make a colourbar for the contour plots
    cb = fig.colorbar(cmap, cax=ax_cbar)
    cb.set_label(input_names[projection_dim] if input_names is not None else None)
    return fig, axes


def _binned_scatter_slice1d(input_data, outputs, dim_x, slice_input_bounds, output_scale="log", ax=None, color="red"):
    """
    Helper function for `slices1d_with_binned_data` that takes the bounds on the input corresponding to a bin,
    and scatter-plots all the data satisfying those bounds.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))  # pragma: no cover

    # Filter the points based on slice_input_bounds
    satisfies_bound = [input_data[:, i] <= slice_input_bounds[i][1] for i in range(input_data.shape[1]) if i != dim_x]
    satisfies_bound.extend(
        [input_data[:, i] >= slice_input_bounds[i][0] for i in range(input_data.shape[1]) if i != dim_x]
    )

    satisfies_all_bounds = functools.reduce(lambda a, b: np.logical_and(a, b), satisfies_bound, True)
    filtered_input_data = input_data[satisfies_all_bounds]
    filtered_outputs = outputs[satisfies_all_bounds]
    ax.scatter(filtered_input_data[:, dim_x], filtered_outputs, color=color, s=50, alpha=0.9)


def plot_slices1d_with_binned_data(
    input_data: np.ndarray,
    outputs: np.ndarray,
    dim_x: int,
    slice_dim1: int,
    slice_dim2: int,
    bounds,
    slice_loc: np.ndarray,
    num_slices: int = 4,
    input_scales: Optional[List[PLOT_SCALE]] = None,
    output_scale: PLOT_SCALE = "linear",
    size: float = 5,
    input_names: Optional[List[str]] = None,
    predict_func: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
) -> plt.Figure:
    """
    Creates a 2D grid with respect to input dimensions slice_dim1 and slice_dim2, and scatter-plots the
    data-points within each bin on a separate sub-plot. If, additionally, `predict_func` is given, slices
    of that function at the mid-point of each bin will also be plotted.

    Example:

    .. image:: ../../tests/data/figures/plot_multidimensional_function_slices_confidence_bounds/figure000.png

    Args:
        input_data (np.ndarray): Array of inputs for the data points to be scatter-plotted.
        outputs (np.ndarray): Array of outputs for the data points to be scatter-plotted.
        dim_x (int): The index of the input dimensions to plot against on the x-axis.
        slice_dim1 (int): Index of the 1st input dimension to bin the data within. Within each row of plots
            the data-points will be within the same bin for dimension indexed by slice_dim1.
        slice_dim2 (int): Index of the 2nd input dimension to bin the data within. Within each column of plots
            the data-points will be within the same bin for dimension indexed by slice_dim2.
        slice_loc: The input coordinates at which to take the slices
        bounds: Array (of shape [num_inputs, 2]) or list of bounds for each input dimension.
            E.g. for two inputs x1 within [0.1, 1.0], x2 within [-1. 2] the bounds would be: [(0.1, 1.), (-1, 2)]
        num_slices (int, optional): How many bins/slices to split the input-range into. As a result,
            num_bins**2 plots will be created, with each subplot corresponding to a unique 2D bin with respect to
            dimensions slice_dim1 and slice_dim2. Defaults to 4.
        input_scales (optional): Scales of each input dimension (log, symlog, linear...). Defaults to linear for all.
        output_scale (optional): Scale of the output dimension. Defaults to "linear".
        size (optional): Size of each slice subplot. Defaults to 3.
        input_names (optional): Names of the input parameters (used for labels). Defaults to None.
        predict_func (optional): If supplied, function to visualise with a 1-D cross-section (1-D slice),
            returning the mean, lower and upper confidence bounds.
            For an input x of shape [N, input_dim], the function should return three arrays of shape [N, 1].
            The slice of this function will be taken at the mid-point of each bin for each sub-plot. Defaults to None.

    Returns:
        The figure
    """
    n_dims: int = len(bounds)
    input_scales = input_scales if input_scales else ["linear"] * n_dims  # type: ignore # auto

    fig, axes = plt.subplots(
        nrows=num_slices, ncols=num_slices, figsize=(size * num_slices, size * num_slices), sharex=True, sharey=True
    )
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    num_boundaries = num_slices * 2 + 1
    slice_dim1_grid = (
        np.geomspace(bounds[slice_dim1][0], bounds[slice_dim1][1], num_boundaries)
        if input_scales[slice_dim1] == "log"
        else np.linspace(bounds[slice_dim1][0], bounds[slice_dim1][1], num_boundaries)
    )
    slice_dim2_grid = (
        np.geomspace(bounds[slice_dim2][0], bounds[slice_dim2][1], num_boundaries)
        if input_scales[slice_dim1] == "log"
        else np.linspace(bounds[slice_dim2][0], bounds[slice_dim2][1], num_boundaries)
    )
    for i in range(num_slices):  # Iterates over the rows of the data.
        slice_dim1_lower = slice_dim1_grid[i * 2]
        slice_dim1_midpoint = slice_dim1_grid[i * 2 + 1]
        slice_dim1_upper = slice_dim1_grid[(i + 1) * 2]
        for j in range(num_slices):  # Iterates over the columns of the data.
            slice_dim2_lower = slice_dim2_grid[j * 2]
            slice_dim2_midpoint = slice_dim2_grid[j * 2 + 1]
            slice_dim2_upper = slice_dim2_grid[(j + 1) * 2]

            slice_input_bounds = bounds.copy()
            slice_input_bounds[slice_dim1] = (slice_dim1_lower, slice_dim1_upper)
            slice_input_bounds[slice_dim2] = (slice_dim2_lower, slice_dim2_upper)

            if predict_func:
                assert slice_loc is not None
                slice_loc_ax = slice_loc.copy()
                slice_loc_ax[slice_dim1] = slice_dim1_midpoint
                slice_loc_ax[slice_dim2] = slice_dim2_midpoint

                plot_1d_slice_through_function_with_confidence_intervals(
                    predict_func,
                    dim=dim_x,
                    slice_loc=slice_loc_ax,
                    slice_bounds=bounds[dim_x],
                    x_scale=input_scales[dim_x],
                    ax=axes[i, j],
                    slice_ind_color=None,
                )
            _binned_scatter_slice1d(input_data, outputs, dim_x, slice_input_bounds=slice_input_bounds, ax=axes[i, j])

            axes[i, j].set_yscale(output_scale)
            axes[i, j].set_xscale(input_scales[dim_x])
            if input_names:  # pragma: no cover
                if i == 0:
                    axes[i, j].set_title(
                        f"{slice_dim2_lower:.3g}$<${input_names[slice_dim2]}$<${slice_dim2_upper:.3g}", fontsize=14
                    )
                elif i == num_slices - 1:
                    axes[i, j].set_xlabel(f"{input_names[dim_x]}", fontsize=9)
        if input_names:  # pragma: no cover
            axes[i, 0].set_ylabel(
                f"{slice_dim1_lower:.3g}$<${input_names[slice_dim1]}$<${slice_dim1_upper:.3g}", fontsize=14
            )
    return fig
