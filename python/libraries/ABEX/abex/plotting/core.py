# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
This file contains generic plotting functions - i.e. ones that can operate on arbitrary data and arbitrary functions
- that can be used as components for more ABEX specific plotting functions (such as plotting the mean prediction,
plotting the simulator, plotting an acquisition function etc.).

The functionality in this file primarily concerns plotting slices, or cross-sections, through a multi-dimensional
function.
"""
from typing import Callable, List, Literal, Optional, Tuple, TypeVar, Union

import matplotlib.colors
import matplotlib.contour
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# The default colour for using lines to mark a point (such as cross-section location). Light red.
LINE_MARKING_COLOUR = "#d13f3f"
PLOT_SCALE = Literal["linear", "symlog", "log"]
NDOrFloatPair = Union[np.ndarray, Tuple[float, float]]


def plot_1d_slice_through_function(
    func: Callable[[np.ndarray], np.ndarray],
    dim: int,
    slice_loc: np.ndarray,
    slice_bounds: NDOrFloatPair,
    ax: Optional[plt.Axes] = None,
    x_scale: PLOT_SCALE = "linear",
    resolution: int = 200,
    color: str = "black",
    slice_ind_color: str = LINE_MARKING_COLOUR,
) -> plt.Axes:
    """Plot 1d slice through the objective function along the dimensions specified by dim, while holding
    the remaining inputs fixed at the values given by slice_loc.

    Example:

    .. image:: ../../tests/data/figures/plot_1d_slice_through_function/figure000.png

    Args:
        func: The function to visualise with a 1-D cross-section (1-D slice).
        dim: dimension (axis) along which to plot a 1D "slice". The inputs to this dimension will vary along the
            x-axis, and the function value will be plotted on the y-axis.
        slice_loc: The coordinates at which the slice is to be taken
        slice_bounds: bounds (min and max values) of the input varied across the slice dimension (dim)
        ax: Axis to plot on. If None, this function creates a new one. Defaults to None.
        x_scale: Scale for the input (log, symlog, linear etc.)
        resolution (optional): How many points to use per dimension for the grid over the slice. Defaults to 200.
        color: Color for the curve
        slice_ind_color: Color to mark the slice location with. Defaults to light red.

    Returns:
        plt.Axes: Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots()

    x_grid, y = calc_1d_slice(
        func=func, dim=dim, slice_loc=slice_loc, slice_bounds=slice_bounds, x_scale=x_scale, resolution=resolution
    )
    # Plot the slice
    ax.plot(x_grid.ravel(), y.ravel(), color=color)
    # Mark the point at which the slice was taken
    ax.axvline(slice_loc[dim], linestyle="--", lw=0.6, color=slice_ind_color, label="Slice Location")
    ax.legend()

    ax.set_xscale(x_scale)
    ax.set_xlim(x_grid[0], x_grid[-1])
    return ax  # type: ignore # auto


def plot_1d_slice_through_function_with_confidence_intervals(
    func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    dim: int,
    slice_loc: np.ndarray,
    slice_bounds: NDOrFloatPair,
    ax: Optional[plt.Axes] = None,
    x_scale: PLOT_SCALE = "linear",
    resolution: int = 200,
    color: str = "black",
    slice_ind_color: Optional[str] = LINE_MARKING_COLOUR,
) -> plt.Axes:
    """Plot 1d slice through a predictive model along the dimensions specified by dim, while holding
    the remaining inputs fixed at the values given by slice_loc. This plotting function takes the given function
    func assumed to return a tri-tuple of mean, lower-bound and upper-bounds predictions, and plots
    the mean with the surrounding lower and upper bounds shaded.

    Example:

    .. image:: ../../tests/data/figures/plot_1d_slice_through_function_with_confidence_intervals/figure000.png

    Args:
        func: The function to visualise with a 1-D cross-section (1-D slice), returning the mean, lower and upper
            confidence bounds. For an input x of shape [N, input_dim], the function should return three arrays
            of shape [N, 1].
        dim: dimension (axis) along which to plot a 1D "slice". The inputs to this dimension will vary along the
            x-axis, and the function value will be plotted on the y-axis.
        slice_loc: The coordinates at which the slice is to be taken
        slice_bounds: bounds (min and max values) of the input varied across the slice dimension (dim)
        ax: Axis to plot on. If None, this function creates a new one. Defaults to None.
        x_scale: Scale for the input (log, symlog, linear etc.)
        resolution (optional): How many points to use per dimension for the grid over the slice. Defaults to 200.
        color: Color for the curve
        slice_ind_color: Color to mark the slice location with. Defaults to light red.

    Returns:
        plt.Axes: Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots()

    x_grid, y = calc_1d_slice(
        func=func, dim=dim, slice_loc=slice_loc, slice_bounds=slice_bounds, x_scale=x_scale, resolution=resolution
    )
    y_mean, y_lb, y_ub = y
    # Plot the confidence interval
    ax.fill_between(x_grid.ravel(), y_lb.ravel(), y_ub.ravel(), color=color, alpha=0.25)
    # Plot the slice
    ax.plot(x_grid.ravel(), y_mean.ravel(), color=color)

    if slice_ind_color:
        # Mark the point at which the slice was taken
        ax.axvline(slice_loc[dim], linestyle="--", lw=0.6, color=slice_ind_color, label="Slice Location")
        ax.legend()

    ax.set_xscale(x_scale)
    ax.set_xlim(x_grid[0], x_grid[-1])
    return ax  # type: ignore # auto


NDAorTuple = TypeVar("NDAorTuple", np.ndarray, Tuple[np.ndarray, ...])


def calc_1d_slice(
    func: Callable[[np.ndarray], NDAorTuple],
    dim: int,
    slice_loc: np.ndarray,
    slice_bounds: NDOrFloatPair,
    x_scale: PLOT_SCALE = "linear",
    resolution: int = 200,
) -> Tuple[np.ndarray, NDAorTuple]:
    """
    Calculate the outputs of a function on a 1d grid that's a slice through the function along the
    dimensions specified by dim, while holding the remaining inputs fixed at those of the point slice_loc.

    For instance, for a function f(x) -> y, where x is D dimensional, and y is 1-dimensional, let's say that
    slice_loc=[5, 9], dim=1 and slice_bounds are [0, 10]. In that case, an input grid will be computed
    along a line between the points:
        [5, 0], [5, 10]
    The value of the function will be computed on that grid and returned together with an array of inputs
    corresponding to that grid.

    Args:
        func: The function to plot. It function has to take arrays of shape [N, D], where N is the number of points
            and return an array (or multiple arrays) of outputs of shape [N, 1].
        dim: Dimension along which to take the 1-D cross-section (if grid used for plotting later-on,
            this corresponds to the x-axis in the plot).
        slice_loc: Location at which to take the slice (the 1-D grid of inputs will be going through this point).
        slice_bounds: bounds (min and max values) of the input varied across the slice dimension (dim)
        x_scale (optional): Scale for the input on the x-axis ("linear", "log"). Defaults to "linear".
        resolution (optional): How many points to use per dimension for the grid over the slice. Defaults to 200.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Returns two arrays: x, y. These two arrays together suffice to plot
            a 1-D slice of the given function. x are the values of the inputs at slice dimensions dim.
            y are the function evaluations at those points.
    """
    xmin, xmax = slice_bounds
    x_grid = np.geomspace(xmin, xmax, resolution) if x_scale == "log" else np.linspace(xmin, xmax, resolution)
    # Specify inputs to the objective function for the 1D slice
    xx = np.tile(slice_loc, (resolution, 1))
    xx[:, dim] = x_grid
    y = func(xx)
    return x_grid, y


def plot_2d_slice_through_function(
    func: Callable[[np.ndarray], np.ndarray],
    dim_x: int,
    dim_y: int,
    slice_loc: np.ndarray,
    slice_bounds_x: NDOrFloatPair,
    slice_bounds_y: NDOrFloatPair,
    ax: Optional[plt.Axes] = None,
    x_scale: PLOT_SCALE = "linear",
    y_scale: PLOT_SCALE = "linear",
    output_scale: PLOT_SCALE = "linear",
    resolution: int = 50,
    slice_ind_color: str = LINE_MARKING_COLOUR,
) -> Tuple[plt.Axes, matplotlib.contour.QuadContourSet]:
    """
    Plot 2d slice through the objective function along the dimensions specified by dim_x, dim_y,
    while holding the remaining inputs fixed at the values given by minimum.

    Example:

    .. image:: ../../tests/data/figures/plot_2d_slice_through_function/figure000.png

    Args:
        func: The function to plot to visualise with a 2-D slice through dimensions dim_x and dim_y.
        dim_x: First dimension along which to plot a 2D "slice", corresponding to the x-axis in the plot.
        dim_y: Second dimension along which to plot a 2D "slice", corresponding to the y-axis in the plot.
        slice_loc: Location at which to take the slices.
        slice_bounds_x: Bounds for the input corresponding to x-axis
        slice_bounds_y: Bounds for the input corresponding to y-axis
        ax (optional): Axis to plot on. Creates a new one if not given. Defaults to None.
        x_scale (optional): Scale for the input on the x-axis. Defaults to "linear".
        y_scale (optional): Scale for the input on the y-axis. Defaults to "linear".
        output_scale (optional): Scale for the output that will be used for the colourmap (log, linear, symlog).
        resolution (optional): How many points to use per dimension for the grid over the slice. Defaults to 50.
        slice_ind_color (optional): Color to mark the slice location with. Defaults to red.

    Returns:
        (plt.Axes, matplotlib.contour.QuadContourSet): Axis with the plot, and the contour set corresponding to
            the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    xx, yy, outputs = calc_2d_slice(
        func=func,
        dim_x=dim_x,
        dim_y=dim_y,
        slice_loc=slice_loc,
        slice_bounds_x=slice_bounds_x,
        slice_bounds_y=slice_bounds_y,
        x_scale=x_scale,
        y_scale=y_scale,
        resolution=resolution,
    )

    _, im = plot_2d_slice_from_arrays(
        xx, yy, outputs, ax=ax, x_scale=x_scale, y_scale=y_scale, output_scale=output_scale
    )

    # Mark the point at which the slice was taken
    mark_point_in_2d(
        ax=ax,  # type: ignore # auto
        x=slice_loc[dim_x],
        y=slice_loc[dim_y],  # type: ignore # auto
        color=slice_ind_color,
        label="Slice Location",
    )  # type: ignore # auto
    return ax, im  # type: ignore # auto


def subspace2d(x0: np.ndarray, dim1: int, dim2: int, x1_points: np.ndarray, x2_points: np.ndarray) -> np.ndarray:
    """Generate an array of inputs with all dimensions being fixed at the values in x0, except for dim1 and dim2, which
    are given by x1_points and x2_points.

    Args:
        x0: An array of shape [ndims]
        dim1: The first dimension to vary along, values in this dimension will be set to values in x1_points
        dim2: The 2nd dimension to vary along, values in this dimension will be set to values in x2_points
        x1_points: Array of shape [n_points]
        x2_points: Array of shaape [n_points]

    Returns:
        Array res of shape [n_points, ndims] where res[:, dim1] = x1_points, res[:, dim2] = x2_points, and all the other
        values correspond to x0
    """
    assert dim1 != dim2, "The dimensions for a 2d subspace can't be the same"
    assert len(x1_points) == len(x2_points)
    xs = np.tile(x0, (len(x1_points), 1))
    xs[:, dim1] = x1_points
    xs[:, dim2] = x2_points
    return np.array(xs)


def calc_2d_slice(
    func: Callable[[np.ndarray], np.ndarray],
    dim_x: int,
    dim_y: int,
    slice_loc: np.ndarray,
    slice_bounds_x: NDOrFloatPair,
    slice_bounds_y: NDOrFloatPair,
    x_scale: PLOT_SCALE = "linear",
    y_scale: PLOT_SCALE = "linear",
    resolution: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the outputs of a function on a 2d grid that's a slice through the function along the
    dimensions specified by dim_x, dim_y, while holding the remaining inputs fixed at those of the point slice_loc.

    For instance, for a function f(x) -> y, where x is D dimensional, and y is 1-dimensional, let's say that
    slice_loc=[5, 7, 9], dim_x=0, dim_y=1 and both slice_bounds_x and slices_bounds_y are [0, 10]. In that case, an
    input grid will be computed on the plane between the 4 corner points:
        [0, 0, 9], [0, 10, 9], [10, 0, 9] and [10, 10, 9]
    The value of the function will be computed on that grid returned together with an input array corresponding to
    that grid

    Args:
        func: The function to plot. It function has to take arrays of shape [N, D], where N is the number of points
            and return an array of outputs of shape [N, 1].
        dim_x: First dimension along which to plot a 2D "slice" (if grid used for plotting later-on,
            this corresponds to the x-axis in the plot).
        dim_y: Second dimension along which to plot a 2D "slice" (if grid used for plotting later-on,
            this corresponds to the y-axis in the plot).
        slice_loc: Location at which to take the slice (the 2D grid of inputs will be going through this point).
        slice_bounds_x: Bounds for the input corresponding to dim_x
        slice_bounds_y: Bounds for the input corresponding to dim_y
        x_scale (optional): Scale for the input on the x-axis ("linear", "log"). Defaults to "linear".
        y_scale (optional): Scale for the input on the y-axis ("linear", "log"). Defaults to "linear".
        resolution (optional): How many points to use per dimension for the 2D grid. Defaults to 50.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Returns three arrays: xx, yy, slice_outputs. These
            three arrays together suffice to contour-plot the 2D-slice of the given function. xx and yy
            are the values of the inputs at dim_x and dim_y respectively for the grid of points, and slice_outputs
            are the function evaluations at those points.
    """
    # Get bounds for the 2 dimensions to plot
    xmin, x_max = slice_bounds_x
    ymin, y_max = slice_bounds_y
    # Generate a grid over dimensions dim_x and dim_y
    x_grid = np.geomspace(xmin, x_max, resolution) if x_scale == "log" else np.linspace(xmin, x_max, resolution)
    y_grid = np.geomspace(ymin, y_max, resolution) if y_scale == "log" else np.linspace(ymin, y_max, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Generate the inputs forming a 2D slice in the input space to the function
    # Input points will be the same as slice_loc except for dimensions dim_x and dim_y
    input_slice_points = subspace2d(slice_loc, dim_x, dim_y, xx.ravel(), yy.ravel())

    slice_outputs = func(input_slice_points)
    slice_outputs = slice_outputs.reshape(xx.shape)
    return xx, yy, slice_outputs


def plot_2d_slice_from_arrays(
    x,
    y,
    z,
    ax: Optional[plt.Axes] = None,
    output_scale: PLOT_SCALE = "linear",
    x_scale: Optional[str] = None,
    y_scale: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    plot_type: str = "pcolormesh",
) -> Tuple[plt.Axes, matplotlib.collections.QuadMesh]:  # type: ignore # auto
    """
    Make a contour plot over the height values specified in z, the coordinates of which are specified in (x, y).

    This function can be used in conjunction with calc_2d_slice() to plot a 2d slice through a function func along
    some specified dimensions. For example:
        >>> x, y, z = calc_2d_slice(
                lambda x: x[:, 0]**2 + x[:, 1]**2, dim_x=0, dim_y=1, slice_loc=np.array([0, 0]),
                slice_bounds_x=(-1, 1), slice_bounds_y=(-1, 1)
            )
        >>> plot_2d_slice_from_arrays(x, y, z)

    Example:

    .. image:: ../../tests/data/figures/plot_2d_slice_from_arrays/figure000.png

    Args:
        x, y: The coordinates of the values in z. Must both be 2-D. They'll both be passed together with z
            to the matplotlib function matplotlib.pyplot.pcolormesh(x, y, z, ...)
        z: The height values over which the contour is drawn. Has to have the same shape as x and y.
        ax (optional): Axis to plot on. Creates a new one if not given. Defaults to None.
        output_scale: Scale of the function output ("log", "symlog", "linear"). Will be used to decide
            what levels to use for the contour plot.
        x_scale (optional): Scale for the input on the x-axis. Defaults to None (no alteration to scale made).
        y_scale (optional): Scale for the input on the y-axis. Defaults to None (no alteration to scale made).
        vmin, vmax: If not None, the range of possible function outputs to use for colour scaling. These values
            will be supplied to the plt.contourf() or plt.pcolormesh()  function call, overriding the default
            color scaling.
        plot_type: Either pcolormesh or contourf

    Returns:
        (plt.Axes, matplotlib.contour.QuadContourSet): Axis with the plot, and the contour set corresponding to
            the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    # Select the right norm for output scale:
    if output_scale == "log":  # pragma: no cover
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)  # type: ignore # auto
    elif output_scale == "symlog":  # pragma: no cover
        norm = matplotlib.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=1.0)  # type: ignore # auto
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # Plot the slice
    if plot_type == "pcolormesh":
        im = ax.pcolormesh(x, y, z, cmap="Greys", norm=norm, shading="gouraud")
    elif plot_type == "contourf":  # pragma: no cover
        im = ax.contourf(x, y, z, cmap="Greys", norm=norm)
    else:  # pragma: no cover
        raise ValueError(f"plot_type {plot_type} not allowed. Must be pcolormesh or contourf.")
    # Scale the axes
    if x_scale:
        ax.set_xscale(x_scale)
    if y_scale:
        ax.set_yscale(y_scale)
    ax.set_xlim(x.ravel()[0], x.ravel()[-1])
    ax.set_ylim(y.ravel()[0], y.ravel()[-1])
    return ax, im  # type: ignore # auto


def mark_point_in_2d(
    ax: plt.Axes, x: float, y: float, color: str = LINE_MARKING_COLOUR, label: Optional[str] = None
) -> None:
    """
    Helper function to mark a point with two crossing lines on a 2D plot. Helpful for clearly marking a specific point
    on contour plots.

    Args:
        ax (plt.Axes): Axis on which to plot the lines through the point.
        x, y: The coordinates of the point to mark.
        color (str, optional): Colour of the lines used to mark the point. Defaults to light red.
    """
    ax.axvline(x, linestyle="--", lw=0.6, color=color, label="Slice Location")
    ax.axhline(y, linestyle="--", lw=0.6, color=color)


def make_lower_triangular_axis_grid_with_colorbar_axes(
    fig: plt.Figure, num_cols: int, num_colorbars: int = 1, hspace: float = 0.15, share_y_on_diagonal: bool = False
) -> Tuple[np.ndarray, List[plt.Axes]]:
    """Helper function to make an axis grid with only lower-triangular part of the grid filled with axes, and
    colourbars residing in the top-right corner. For instance, for 3 columns and a single colourbar, the layout
    would look roughly as follows:

        [x] [ ] |c|
        [x] [x] |c|
        [x] [x] [x]

    Where [x] indicates axes for plots, and the two |c| cells represent a single colourbar axis on which
    a vertical colorbar can be plotted. For more than one colourbar, these will be stacked side by side, occupying
    one column of space.

    Args:
        fig (plt.Figure): Figure for which to make the axes.
        num_cols (int): Number of columns (and rows) of axes in the grid.
        num_colorbars (int, optional): Number of colourbars to make axes for. Defaults to 1.
        hspace (float, optional): The parameter controlling horizontal space between axes. Defaults to 0.15.
        share_y_on_diagonal (bool): Whether to share the y-axis between the diagonal plots (useful if diagonal plots
            show 1d slices through function)

    Returns:
        Tuple[np.ndarray, List[plt.Axes]]: The first return element is an array with lower-triangular entries
            corresponding to the created axes on which to plot (the remaining entries are None). The second
            returned element contains a list of axis of length num_colorbars for the colourbars.
    """
    # assert num_cols > 1
    colorbar_margin = 0.15 if num_colorbars >= 2 else 0.7
    colorbar_axis_width = (1 - colorbar_margin - hspace) / num_colorbars

    cbar_axes = []

    if num_cols >= 2:
        # Construct grid-spec
        spec = gridspec.GridSpec(
            ncols=num_cols + num_colorbars,
            nrows=num_cols,
            figure=fig,
            width_ratios=(
                [1] * (num_cols - 1) + [colorbar_margin] + [colorbar_axis_width] * num_colorbars  # type: ignore
            ),  # type: ignore
        )
        # Make colorbar axes
        for i in range(num_colorbars, 0, -1):
            cbar_ax = fig.add_subplot(spec[:-1, -i])
            cbar_axes.append(cbar_ax)
    else:  # pragma: no cover
        spec = gridspec.GridSpec(
            ncols=2 + num_colorbars,
            nrows=num_cols,
            figure=fig,
            width_ratios=[1] + [0.15] + [colorbar_axis_width] * num_colorbars,  # type: ignore
        )
        # Make colorbar axes
        for i in range(num_colorbars, 0, -1):
            cbar_ax = fig.add_subplot(spec[:, -i])
            cbar_axes.append(cbar_ax)

    spec.update(wspace=0.25, hspace=hspace)
    axes = np.empty([num_cols, num_cols], dtype=object)
    # Create a grid of axes from the grid-spec
    for i in range(num_cols):  # i iterates over the rows of the plots
        for j in range(num_cols):  # j iterates over the columns of the plots
            if j == num_cols - 1:
                grid_slice = spec[i, j:]
            else:
                grid_slice = spec[i, j]
            if i == j:
                if i == 0 or not share_y_on_diagonal:
                    axes[i, j] = fig.add_subplot(grid_slice)
                else:
                    axes[i, j] = fig.add_subplot(grid_slice, sharey=axes[i - 1, j - 1])
            elif i > j:
                if j > 0:
                    # If j > 0, share the y axis with the axis to the left
                    axes[i, j] = fig.add_subplot(grid_slice, sharex=axes[i - 1, j], sharey=axes[i, j - 1])
                else:
                    axes[i, j] = fig.add_subplot(grid_slice, sharex=axes[i - 1, j])
    return axes, cbar_axes
