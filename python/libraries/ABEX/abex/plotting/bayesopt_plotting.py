# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Tuple, Union

# Avoid spurious X windows errors, see:
# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import GPy
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from abex.constants import FILE
from abex.dataset import Dataset
from abex.plotting.composite_core import plot_multidimensional_function_slices
from abex.plotting.core import (
    PLOT_SCALE,
    calc_2d_slice,
    make_lower_triangular_axis_grid_with_colorbar_axes,
    plot_2d_slice_from_arrays,
)
from azureml.core import Run
from matplotlib.axes import Axes
from psbutils.type_annotations import PathOrString

if TYPE_CHECKING:
    # Imports BayesOptModel for static type-checks only to get around circular import problem
    from abex.bayesopt import BayesOptModel  # pragma: no cover

RELATIVE_VALUE = "Relative value"
# noinspection PyUnresolvedReferences
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # type: ignore


def loss_convergence(losses: List[List[float]], fname: Optional[str] = None) -> None:  # pragma: no cover
    f = plt.figure(figsize=(6, 4))
    for i, loss in enumerate(losses):
        iterations = len(loss)
        plt.scatter(list(range(iterations)), loss, label=f"Fold {i}", s=3)
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Marginal log-likelihood")
    sns.despine()
    if fname is not None:
        f.savefig(fname, bbox_inches="tight")
        # noinspection PyArgumentList
        plt.close()


def opt_distance(X, optx, j):  # pragma: no cover
    # noinspection PyUnresolvedReferences
    Nd, n_inputs = np.shape(X)
    Ij = np.eye(n_inputs)
    Ij[j, j] = 0
    d = np.zeros(Nd)
    for i in range(Nd):
        xd = X[i, :] - optx
        d[i] = xd @ Ij @ xd
    return d


def simulation_panel1d(
    ax: Axes,
    predict_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    slice_dim: int,
    bounds: Tuple[float, float],
    slice_x: np.ndarray,
    slice_y: Optional[float] = None,
    resolution: int = 101,
    color="b",
) -> Axes:  # pragma: no cover
    """Make a plot of the predicted output (and +- one standard deviation) against one input (defined by slice_dim)
    that is a slice through input space at a location defined by slice_x.

    Optionally, mark the point [slice_x, slice_y], which, if the slice is plotted at a maximum of the model's
    predictions, would be the maximum of the model predictions.
    """
    # Get a grid of inputs for the continuous variable being varied across this slice
    x_grid = np.linspace(bounds[0], bounds[1], resolution)
    xs = np.tile(slice_x, (len(x_grid), 1))
    xs[:, slice_dim] = x_grid
    y_pred, y_var = predict_func(xs)
    sigma = np.sqrt(y_var)
    ax.plot(x_grid, y_pred, "-", label="Prediction", c=color)
    ax.fill_between(
        x_grid,
        y_pred - sigma,
        y_pred + sigma,
        alpha=0.25,
        fc=color,
        ec="None",
        label="68% confidence interval",
    )
    ax.set_xlim(bounds[0], bounds[1])
    if slice_y is not None:
        ax.plot(slice_x[slice_dim], slice_y, "o", markeredgecolor=color, markerfacecolor="w", label="Optimum")
    else:
        ax.axvline(slice_x[slice_dim], alpha=0.2, linestyle="--")
    return ax


def get_logged_img_title(title: Optional[str] = None, fname: Optional[PathOrString] = None) -> str:
    """
    Creates a title for logging plots on AML. If title is provided, that forms the base, otherwise use default.
    If filename is provided, append the filename, which contains information about iteration and seed number.
    Args:
        title:
        fname:

    Returns:

    """
    title = title or "plot"
    if fname is not None:
        assert isinstance(fname, Path)
        title += f"_{fname.stem}"
    return title


def plot_prediction_slices1d(
    predict_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    parameter_space: "OrderedDict[str, Tuple[float, float]]",
    slice_loc: np.ndarray,
    slice_y: Optional[float] = None,
    scatter_x: Optional[np.ndarray] = None,
    scatter_y: Optional[np.ndarray] = None,
    output_label: str = "Objective",
    resolution: int = 100,
    size: int = 3,
    num_cols: Optional[int] = None,
    title: Optional[str] = None,
    fname: Optional[PathOrString] = None,
) -> Tuple[plt.Figure, np.ndarray]:  # pragma: no cover
    """

    Plot slices of the predictions from the model crossing a given location.

    Args:
        predict_func (Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]): A function taking an input and returning
            mean and variance of the predictive distribution at those points.
        parameter_space (OrderedDict[str, Tuple[float, float]]): An ordered dictionary mapping input names to bounds.
        slice_loc (np.ndarray): The point through which to plot the slices of the predictive distribution.
        slice_y (Optional[float], optional): The output value the the slice location. Defaults to None.
        scatter_x (Optional[np.ndarray], optional): Points which to scatter on the plot (project onto the slices).
            If given, scatter_y must be specified as well. Defaults to None.
        scatter_y (Optional[np.ndarray], optional): Output values corresponding to scatter_x. Defaults to None.
        output_label (str, optional): Label for the output axis. Defaults to "Objective".
        resolution (int, optional): Resolution (num. points) for the grid of input points along each slice.
            Defaults to 100.
        size (int, optional): Size of each axis with one slice in inches. Defaults to 3.
        num_cols (Optional[int], optional): Maximum number of columns. If more slices, the axes will wrap.
            Defaults to num_cols = ceil(sqrt(num_input_dims)).
        title (Optional[str], optional): Title for the plot. Defaults to None.
        fname (Optional[PathOrString], optional): File-name where to save the plot. Defaults to None.
    """
    parameters = list(parameter_space.items())
    n_inputs = len(parameter_space)
    num_cols = num_cols if num_cols else int(np.ceil(np.sqrt(n_inputs)))
    num_rows = int(np.ceil(n_inputs / num_cols))
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        sharey=True,
        figsize=(size * num_cols, size * num_rows),
    )
    axs = np.atleast_2d(axs)  # type: ignore
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i, j]
            slice_dim = i * num_cols + j
            if slice_dim < n_inputs:
                param_name, bounds = parameters[slice_dim]
                simulation_panel1d(
                    ax=ax,
                    predict_func=predict_func,
                    slice_x=slice_loc,
                    slice_y=slice_y,
                    bounds=bounds,
                    slice_dim=slice_dim,
                    color=colors[0],
                    resolution=resolution,
                )

                # Scatter-plot data points if points to scatter given
                if scatter_x is not None and scatter_y is not None:
                    ax.scatter(scatter_x[:, slice_dim], scatter_y, s=3, c=colors[1])
                ax.set_xlabel(param_name)
            else:
                ax.set_visible(False)
        axs[i, 0].set_ylabel(output_label)

    if title is not None:
        fig.suptitle(title)
    # noinspection PyUnresolvedReferences
    plt.tight_layout()
    sns.despine()

    run = Run.get_context()
    logged_img_title = get_logged_img_title(title="plot1d", fname=fname)
    run.log_image(name=logged_img_title, plot=plt)
    # If filename given, save
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
        # noinspection PyArgumentList
        plt.close()

    return fig, axs


def plot_prediction_slices2d(
    predict_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    parameter_space: "OrderedDict[str, Tuple[float, float]]",
    slice_loc: np.ndarray,
    scatter_x: Optional[np.ndarray] = None,
    scatter_y: Optional[np.ndarray] = None,
    output_label: Optional[str] = None,
    resolution: int = 100,
    size: int = 3,
    title: Optional[str] = None,
    fname: Optional[PathOrString] = None,
) -> Tuple[plt.Figure, np.ndarray]:  # pragma: no cover
    parameters = list(parameter_space.items())
    n_inputs = len(parameters)
    assert n_inputs >= 2, "At least two input dimensions are required to plots 2d slice"

    # Keep a running minimum and maximum of function values in 2D slices
    func_values_min, func_values_max = np.inf, -np.inf
    # Keep track of contour sets returned for each axis
    contour_sets = []

    num_cols = n_inputs - 1  # Number of rows of axes equals number of columns
    # Construct axes
    # noinspection PyTypeChecker
    fig = plt.figure(figsize=(size * num_cols, size * num_cols))
    axes, cbar_axes = make_lower_triangular_axis_grid_with_colorbar_axes(fig=fig, num_cols=num_cols, num_colorbars=1)
    for i in range(num_cols):  # i iterates over the rows of the plots
        y_param_dim = i + 1
        y_param_name, y_bounds = parameters[y_param_dim]
        for j in range(num_cols):  # j iterates over the columns of the plots
            ax = axes[i, j]
            if j <= i:
                # Indices of the inputs to plot
                x_param_dim = j
                x_param_name, x_bounds = parameters[x_param_dim]
                # Compute the data for the 2D slice plot
                xx, yy, func_values_slice = calc_2d_slice(
                    func=lambda x: predict_func(x)[0],  # Only interested in the mean of the prediction
                    dim_x=x_param_dim,
                    dim_y=y_param_dim,
                    slice_loc=slice_loc,
                    slice_bounds_x=x_bounds,
                    slice_bounds_y=y_bounds,
                    resolution=resolution,
                )
                # Plot the 2D slice
                _, contour_set = plot_2d_slice_from_arrays(xx, yy, func_values_slice, ax=ax, plot_type="contourf")
                contour_sets.append(contour_set)

                # Keep a running minimum and maximum of function values in slices
                func_values_min = min(func_values_min, func_values_slice.min())  # type: ignore
                func_values_max = max(func_values_max, func_values_slice.max())  # type: ignore

                # Scatter-plot the data
                if scatter_x is not None and scatter_y is not None:
                    if len(scatter_y) > 0:
                        s = (scatter_y - np.min(scatter_y)) / np.max(scatter_y) + 1
                        ax.scatter(scatter_x[:, x_param_dim], scatter_x[:, y_param_dim], s=5 * s, c="yellow")
                ax.set_xlim(x_bounds[0], x_bounds[1])
                ax.set_ylim(y_bounds[0], y_bounds[1])
                if i == num_cols - 1:
                    ax.set_xlabel(x_param_name)
                else:
                    # Remove redundant ticks on inner plots
                    ax.xaxis.set_visible(False)
                if j > 0:
                    ax.yaxis.set_visible(False)

        axes[i, 0].set_ylabel(y_param_name)
    # Update norm limits for colour scaling for each axis:
    for im in contour_sets:
        im.set_clim(vmin=func_values_min, vmax=func_values_max)

    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    cb = fig.colorbar(contour_sets[-1], cax=cbar_axes[0])
    cb.set_label(output_label)
    cbar_axes[0].yaxis.set_ticks_position("left")

    if title is not None:
        fig.suptitle(title)

    run = Run.get_context()

    logged_img_title = get_logged_img_title(title="plot2d", fname=fname)
    run.log_image(name=logged_img_title, plot=plt)

    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
        # noinspection PyArgumentList
        plt.close()
    return fig, axes


def plot_calibration_curve(
    predict_func: Callable, datasets: List[Dataset], labels: List[str]
) -> Tuple[plt.Figure, plt.Axes]:  # pragma: no cover
    """Plot a calibration curve - the curve showing the percentage of points within each confidence interval around
    the mean prediction from the model. This is useful for gauging how reliable uncertainty estimates from the model
    are.

    Args:
        predict_func (Callable): A function taking an array of inputs and returning a tuple of two arrays: mean
            prediction and variance of the predictive distribution (which is assumed to be Gaussian).
        datasets (List[Dataset]): A list of datasets for which to plot calibration curves.
        labels (List[str]): A list of labels for each dataset of the same length as datasets.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    with sns.axes_style("whitegrid"):
        # Plot the individual calibration curves for each dataset
        for i, dataset in enumerate(datasets):
            _make_single_calibration_curve(predict_func, dataset.inputs_array, dataset.output_array, ax, labels[i])

        plt.plot([0, 1], [0, 1], ":", color="gray", alpha=0.3, label="Ideal", zorder=-1)
        ax.set_xlabel("Predictive Confidence Interval")
        ax.set_ylabel("Percentage points within that interval")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.legend()
        sns.despine()
    return fig, ax


def _make_single_calibration_curve(
    predict_func: Callable, x: np.ndarray, y: np.ndarray, ax: Optional[plt.Axes] = None, label: Optional[str] = None
) -> plt.Axes:  # pragma: no cover
    if ax is None:
        fig, ax = plt.subplot(figsize=(5, 5))  # type: ignore
        assert ax is not None
    mean, variances = predict_func(x)
    stds = np.sqrt(variances)
    residuals = np.abs(y - mean)
    # Normalised residuals is the number of standard deviations the observed point is away from mean prediction
    normalised_residuals = residuals / stds
    normalised_residuals = np.sort(normalised_residuals, axis=0).ravel()
    # Convert num. of standard deviations from mean to confidence interval (centered around the mean)
    confidence = scipy.stats.norm.cdf(normalised_residuals) - scipy.stats.norm.cdf(  # type: ignore
        -normalised_residuals
    )
    confidence = np.insert(confidence, 0, 0.0)  # Insert a (0% confidence, 0% points) entry at start
    # Percent points observed
    perc_points = np.linspace(0, 1, len(confidence))
    # Append the end-point of the curve (100% confidence, 100% points observed)
    confidence = np.append(confidence, [1.0])
    perc_points = np.append(perc_points, [1.0])
    plt.step(confidence, perc_points, where="post", label=label, alpha=0.75, linewidth=3.0)
    return ax


# noinspection PyTypeChecker
def _decorate_axis_predicted_against_observed(model: "BayesOptModel", ax: Axes) -> None:  # pragma: no cover
    """Label and set limits of axes for a predictions against observed scatter-plots."""
    ax.set_xlabel(f"Observed: {model.train.transformed_output_name}")
    ax.set_ylabel(f"Predicted: {model.train.transformed_output_name}")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lims = np.array([np.minimum(xlim[0], ylim[0]), np.maximum(xlim[1], ylim[1])])
    ax.plot(lims, lims, "k--")
    ax.fill_between(lims, lims - np.log10(2.0), lims + np.log10(2.0), color=(0.7, 0.7, 0.7), zorder=-1)
    ax.fill_between(lims, lims - np.log10(3.0), lims + np.log10(3.0), color=(0.85, 0.85, 0.85), zorder=-2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def plot_train_test_predictions(
    model: "BayesOptModel",
    category: Optional[str] = None,
    ms: int = 10,
    alpha: float = 0.25,
    output_path: Optional[PathOrString] = None,
) -> None:  # pragma: no cover
    """Plot predicted outputs against the actual observed outputs for the train-set and the test-set for this model.
    If category is given, plot the points in different colour depending on the category.

    Args:
        model: Model to plot predictions of
        category (optional): Which category to condition on. Points will be plotted with different colour
            depending on the value of that category. Defaults to None (don't condition on any category).
        ms (optional): Marker size. Defaults to 10.
        alpha (optional): Opacity of plotted points. Defaults to 0.25.
        output_path (optional): Path where to save the plot. Defaults to None.

    TODO: There is an argument to be made that these plots should be made in the original,
    rather than dataset, space if possible. One might want to compare the performance of the models from the plots
    when different pre-processing steps are used for instance.
    """
    axs: List[Axes]
    # noinspection PyTypeChecker
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))  # type: ignore
    # Add the plot for train points on the first axis
    train_labels = model.train.categorical_inputs_df[category] if category else None
    _add_errorbar_predictions_against_observed(
        ax=axs[0], model=model, dataset=model.train, labels=np.asarray(train_labels), ms=ms, alpha=alpha
    )
    # Add the plot for test points on the second axis
    assert model.test is not None
    test_labels = model.test.categorical_inputs_df[category] if category else None
    _add_errorbar_predictions_against_observed(
        ax=axs[1], model=model, dataset=model.test, labels=np.asarray(test_labels), ms=ms, alpha=alpha
    )
    if category is not None:
        axs[0].legend()
    for ax in axs:
        _decorate_axis_predicted_against_observed(model, ax)

    axs[0].set_title(f"Train ($r = {model.r_train:.3f}$)")
    axs[1].set_title(f"Test ($r = {model.r_test:.3f}$)")
    sns.despine()
    # noinspection PyUnresolvedReferences
    plt.tight_layout()  # type: ignore
    if output_path is not None:
        f.savefig(output_path, bbox_inches="tight")
        # noinspection PyArgumentList
        plt.close()  # type: ignore


def _add_errorbar_predictions_against_observed(
    ax: Axes, model: "BayesOptModel", dataset: Dataset, labels: Optional[np.ndarray], ms: float, alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover
    """Helper function to add errorbar plots of predicted outputs against the actual observed outputs on a given
    dataset. The plot is added onto an axis give by argument ax.
    """
    # Get the mean and variance of surrogate model predictions
    y_mean, y_var = model.minus_predict(dataset.inputs_array)
    y_std = np.sqrt(y_var)
    # Get the true (observed) outputs
    y_obs = dataset.output_array
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            locs = np.where(labels == label)
            ax.errorbar(y_obs[locs], y_mean[locs], y_std[locs], fmt=".", ms=ms, alpha=alpha, label=label)
    else:
        ax.errorbar(y_obs, y_mean, y_std, fmt=".", ms=ms, alpha=alpha)  # pragma: no cover
    return y_mean, y_std, y_obs


def plot_predictions_against_observed(
    ax: Axes,
    models: List["BayesOptModel"],
    datasets: List[Dataset],
    title: str,
    category: Optional[str] = None,
    by_file: bool = False,
    ms: float = 10,
    alpha: float = 0.25,
) -> float:  # pragma: no cover
    """
    Plot predicted outputs against the actual observed outputs for the datasets given (which could be the corresponding
    test-sets of each of the cross-validation models).

    If "category" is given, plot the points in different colour depending on the value of the categorical variable
    "category".

    Args:
        ax: Axis to plot on
        models: List of models to corresponding to each cross-validation fold
        datasets: A list of same length as models with the corresponding datasets to evaluate each model on.
        title: Title for the plot (e.g. "Cross-validation")
        category (optional): Which category to condition on. Points will be plotted with different colour
            depending on the value of that category. Defaults to None (don't condition on any category).
        by_file (bool): Points will be plotted with colours reflecting the FILE identifier.
        ms (optional): Marker size. Defaults to 10.
        alpha (optional): Opacity of plotted points. Defaults to 0.25.

    Returns:
        Pearson correlation coefficient (combined for all folds)
    """
    Y_pred, Y_obs = np.empty((0,)), np.empty((0,))
    legend_title: Optional[str] = None
    for model, dataset in zip(models, datasets):
        if by_file:  # pragma: no cover
            labels = dataset.file
            legend_title = FILE
        elif category is not None:
            labels = np.asarray(dataset.categorical_inputs_df[category])
            legend_title = category
        else:  # pragma: no cover
            labels = None
            legend_title = None
        y_mean_test, _, y_obs_test = _add_errorbar_predictions_against_observed(
            ax=ax, model=model, dataset=dataset, labels=labels, ms=ms, alpha=alpha
        )
        Y_pred = np.append(Y_pred, y_mean_test)
        Y_obs = np.append(Y_obs, y_obs_test)
    if category:
        ax.legend(title=legend_title)
    # Compute Pearson's correlation
    r = float(np.corrcoef(Y_pred, Y_obs)[1, 0])

    _decorate_axis_predicted_against_observed(models[0], ax)
    ax.set_title(f"{title} ($r = {r:.3f}$)")
    # noinspection PyUnresolvedReferences
    plt.tight_layout()  # type: ignore
    sns.despine(ax=ax)
    return r


def hmc_traces(burnin: Axes, samples: Axes, fname: Optional[str] = None):  # pragma: no cover
    # noinspection PyTypeChecker
    f, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    sample = "Sample"
    burnin.plot(x=sample, ax=axs[0], title="Burn-in")
    samples.plot(x=sample, ax=axs[1], title="Samples")
    sns.despine()
    if fname is not None:
        f.savefig(fname, bbox_inches="tight")
        # noinspection PyArgumentList
        plt.close()


def hmc_samples(samples_dataframe: pd.DataFrame, fname: Optional[PathOrString] = None) -> None:
    """Visualise the samples of the parameters of a model with a collection of pair-wise scatter-plots.

    Args:
        samples_dataframe (pd.DataFrame): A DataFrame with each row of values corresponding the a single HMC sample
            (these values can represent the parameter values, or log-likelihood of that sample for instance).
        fname (Optional[PathOrString], optional): If given, where to save the plot. Defaults to None.
    """
    f = sns.pairplot(samples_dataframe, diag_kind="kde")
    sns.despine()
    if fname is not None:
        f.savefig(fname, bbox_inches="tight")
        # noinspection PyArgumentList
        plt.close()


def distance(x: np.ndarray) -> np.ndarray:  # pragma: no cover
    n = len(x)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            # noinspection PyUnresolvedReferences
            d[i, j] = np.linalg.norm(x[i] - x[j])
            d[j, i] = d[i, j]
    return d


# noinspection PyUnresolvedReferences
def experiment_distance(expt: np.ndarray, fname: Optional[str] = None) -> None:  # pragma: no cover
    n = np.shape(expt)[0]
    f = plt.figure(figsize=(4.5, 4))
    xx, yy = np.meshgrid(list(range(n)), list(range(n)))
    d = distance(expt)
    # TODO: d is an array of float, but c should be a list of color names.
    # noinspection PyTypeChecker
    plt.scatter(x=xx, y=yy, c=d, cmap="jet")  # type: ignore
    plt.gca().invert_yaxis()
    experiment_id = "Experiment ID"
    plt.xlabel(experiment_id)
    plt.ylabel(experiment_id)
    plt.colorbar(label="Euclidean distance")
    sns.despine()
    if fname is not None:
        f.savefig(fname, bbox_inches="tight")
        # noinspection PyArgumentList
        plt.close()


def plot_pred_objective_for_batch(
    batch: pd.DataFrame,
    predict_func: Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    bounds: "OrderedDict[str, Tuple[float, float]]",
    dataset: Optional[Dataset] = None,
    columns_to_plot: Optional[List[str]] = None,
    units: Optional[List[Union[str, None]]] = None,
    num_cols: int = 4,
    input_scales: Optional[List[str]] = None,
    output_scale: str = "linear",
    fname: Optional[Path] = None,
    output_label: str = "Output",
    subplot_size: float = 3,
) -> None:  # pragma: no cover
    """Plot the batch of points generated from the bayesopt procedure against each input dimension, together
    with the objective value prediction from the model.

    Args:
        batch: DataFrame with the batch of inputs to plot model predictions for (could be in either the model input
            space, or original "pretransform" input space)
        predict_func: Function which returns the mean and lower and upper confidence bounds for the prediction at each
            point. This is helpful if preprocessing needs to be applied to data before passing to a model.
        bounds:
            Constraints on the input-space for the Bayes. Opt. procedure to visualise.
        dataset: Experiment data to plot alongside. Currently only plots in the original input space
        num_cols: Maximum number of columns in the plot
        input_scales (optional): Scales of each input dimension (log, symlog, linear...). Defaults to linear for all.
        output_scale (optional): Scale of the output dimension. Defaults to "linear".
        fname: Path to where to save the plot. Don't save it if None. Defaults to None.
        output_label: The label for the y-axis. Defaults to 'Output'.
        subplot_size: The size in inches of each individual subplot
    """
    # Get model predictions at the experiment batch locations (mean, lower confidence bound, upper confidence bound)
    mean_pred, lb_pred, ub_pred = predict_func(batch)
    y_error = (mean_pred - lb_pred, ub_pred - mean_pred)

    # If input_scales not specified, default all to 'linear'
    input_scales = input_scales if input_scales else ["linear"] * len(bounds)

    # Get the names of variables to plot
    columns_to_plot = columns_to_plot if columns_to_plot else batch.columns  # type: ignore
    assert columns_to_plot is not None
    units = units if units else [None] * len(columns_to_plot)  # type: ignore
    n_inputs = len(columns_to_plot)
    num_rows = int(np.ceil(n_inputs / num_cols))
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(
        nrows=num_rows, ncols=num_cols, sharey=True, figsize=(subplot_size * num_cols, subplot_size * num_rows)
    )
    # Ensure that axs are a 2d grid of axes, even if num_rows=1
    axs = np.atleast_2d(axs)  # type: ignore
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i, j]
            input_idx = i * num_cols + j
            if input_idx < n_inputs:
                col = columns_to_plot[input_idx]
                # Plot model predictions
                ax.errorbar(
                    batch[col].values,
                    mean_pred,
                    yerr=y_error,
                    fmt="o",
                    color=colors[1],
                    markerfacecolor="w",
                    label="Model",
                )
                # Plot data in dataset
                if dataset:
                    # TODO: This bit currently assumes that the plotting happens in pretransformed space.
                    dataset_x = dataset.pretransform_df[col].values
                    dataset_y = dataset.pretransform_df[dataset.pretransform_output_name].values
                    ax.scatter(dataset_x, dataset_y, s=3, c=colors[0], label="Data")

                # Set axis scales
                ax.set_yscale(output_scale)
                ax.set_xscale(input_scales[input_idx])

                axlabel = col + f" ({units[input_idx]})" if units[input_idx] else col
                ax.set_xlabel(axlabel)
                ax.axvspan(*bounds[col], ymax=0.1, color=colors[1], alpha=0.3, label="Bounds")
            else:
                ax.set_visible(False)
        axs[i, 0].set_ylabel(output_label)

    # noinspection PyUnresolvedReferences
    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc=[0.82, 0.2])

    # noinspection PyUnresolvedReferences
    plt.tight_layout()
    sns.despine()
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
        # noinspection PyArgumentList
        plt.close()


def acquisition1d(
    model: "BayesOptModel",
    x0: np.ndarray,
    is_input_normalised: bool = True,
    num_cols: int = 5,
    num_xs: int = 101,
    title: Optional[str] = None,
    fname: Optional[Path] = None,
) -> None:  # pragma: no cover
    """
    Plot the acquisition function in 1d variations around a reference point

    Args:
        model: The model
        x0 (numpy array): The reference point
        is_input_normalised (bool): Whether the input data are normalised
        num_cols (int): Number of columns in subplot grid
        num_xs (int): Number of grid-points to calculate the input axis
        title (str): Figure title
        fname (str): Optional file path to save figure
    """
    y0 = model.acquisition.evaluate(x0[np.newaxis])  # type: ignore
    parameters = list(model.continuous_parameters.items())
    n_inputs = len(parameters)
    num_rows = int(np.ceil(n_inputs / num_cols))
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(
        num_rows, num_cols, sharex=is_input_normalised, sharey=True, figsize=(3 * num_cols, 3 * num_rows)
    )
    # Ensure that axs are a 2d grid of axes, even if num_rows is 1
    axs = np.atleast_2d(axs)  # type: ignore
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i][j]
            input_idx = i * num_cols + j
            if input_idx < n_inputs:
                pname, bounds = parameters[input_idx]
                # make a grid of inputs along the input_idx dimension while keeping the other values same as in x0
                cs = np.linspace(0, bounds[1], num_xs)
                xs = np.tile(x0, (num_xs, 1))
                xs[:, input_idx] = cs
                ax.plot(cs, model.acquisition.evaluate(xs), "-")  # type: ignore
                ax.plot(x0[input_idx], y0.ravel(), marker="o", markerfacecolor="w", label="Model optimum")

                if is_input_normalised:
                    ax.set_title(pname)
                    if i == num_rows - 1:
                        ax.set_xlabel(RELATIVE_VALUE)
                else:
                    ax.set_xlabel(pname)
            else:
                ax.set_visible(False)
        axs[i][0].set_ylabel("Acquisition")

    if title is not None:
        fig.suptitle(title)
    else:
        # noinspection PyUnresolvedReferences
        plt.tight_layout()
    sns.despine()

    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)


def plot_acquisition_slices(
    model: "BayesOptModel",
    dataset: Dataset,
    slice_loc: np.ndarray,
    input_names: Optional[List[str]] = None,
    input_scales: Optional[List[PLOT_SCALE]] = None,
    onehot_context: Optional[Iterable[float]] = None,
    output_label: str = "Acquisition Value",
    fname: Optional[Path] = None,
) -> None:  # pragma: no cover
    """
    Plot 2d acquisition function slices through the input space. A wrapper
    around plot_multidimensional_function_slices().

    Args:
        model: Model for which to plot slices through acquisition function.
        dataset: Dataset with the preprocessing transform to apply to inputs.
        slice_loc: Location at which to take slices in original space.
        input_names: Names of the input variables in original space.
        input_scales: Plotting scales for the inputs (linear, log, etc.).
        onehot_context: If given, the onehot encoding of the categorical variables on which to condition
            the slices (i.e. in all slices the input for categorical variables will be fixed to that value)
        output_label: Label for the acquisition function output (can be acqusition function name).
        fname: If given, the plot will be saved there.
    """

    def acquisition_with_preprocessing(x: np.ndarray):
        assert model.acquisition is not None
        x_trans = dataset.preprocessing_transform(pd.DataFrame(x, columns=input_names))
        x_trans = x_trans[dataset.transformed_input_names].values
        if onehot_context:
            x_trans = np.concatenate((x_trans, np.tile(onehot_context, [x_trans.shape[0], 1])), axis=1)
        return model.acquisition.evaluate(x_trans)  # type: ignore # auto

    fig, _ = plot_multidimensional_function_slices(
        func=acquisition_with_preprocessing,
        slice_loc=slice_loc,
        bounds=list(dataset.pretransform_cont_param_bounds.values()),
        input_names=input_names,
        input_scales=input_scales,
        output_scale="linear",
        output_label=output_label,
    )

    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    return


def plot_gpy_priors(
    priors: List[GPy.priors.Prior], param_names: List[str], size: float = 3, allow_logspace: bool = False
) -> plt.Figure:  # pragma: no cover
    """Visualise a set of priors on parameters (corresponding, for instance, to model parameters) by plotting
    their density.

    Args:
        priors (List[GPy.priors.Prior]): A list of GPy.priors.Prior objects the PDFs of which will be drawn.
        param_names (List[str]): List of names of parameters corresponding to the priors.
        size (float, optional): Size (height) of each sub-plot in inches. Defaults to 3.
        allow_logspace (bool, optional): Whether to plot a parameter in log-space if the parameter is strictly
            positive. Defaults to False.

    Returns:
        plt.Figure: The figure
    """
    assert len(param_names) == len(priors)
    num_params = len(param_names)
    fig, axes = plt.subplots(ncols=num_params, figsize=(1.3 * size * num_params, size))
    colors = sns.color_palette("pastel", num_params)

    with sns.axes_style("whitegrid"):
        for i, (param_name, prior) in enumerate(zip(param_names, priors)):
            samples = prior.rvs(1000)  # type: ignore
            xmin, xmax = np.percentile(samples, 0.1), np.percentile(samples, 99.9)  # Remove outliers
            if samples.min() > 0 and allow_logspace:
                axes[i].set_xscale("log")
                bins = np.geomspace(xmin, xmax, 6)
                x_grid = np.geomspace(xmin, xmax, 100)
            else:
                bins = np.linspace(xmin, xmax, 50)
                x_grid = np.linspace(xmin, xmax, 100)
            axes[i].hist(samples, bins=bins, density=True, alpha=0.7, color=colors[i])
            axes[i].plot(x_grid, prior.pdf(x_grid), linewidth=2.0, color=colors[i])
            # Mark mean and +- standard deviation
            samples_mean, samples_std = samples.mean(), samples.std()
            axes[i].axvline(samples_mean, color="black", zorder=-1, label="Mean")
            axes[i].axvspan(
                samples_mean - samples_std,
                samples_mean + samples_std,
                color="gray",
                zorder=-1,
                alpha=0.2,
                hatch="/",
                label="Mean$\\pm$std.",
            )
            axes[i].set_title(param_name, fontsize=12)
            axes[i].set_xlabel("$x$")
            axes[i].set_xlim(xmin, xmax)
            plt.legend()
        axes[0].set_ylabel("$p(x)$")
        plt.subplots_adjust(wspace=0.2)
    return fig
