# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from itertools import cycle
from typing import Callable, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
from functools import partial


def _plot_running_best_trace_distribution(
    objectives_per_experiment: np.ndarray,  # Shape [num_experiments, num_steps]
    color,
    label: str,
    ax: plt.Axes,
    linestyle: Literal["-", "--", ":", "-."] = "-",
    confidence_interval: Optional[float] = None,
    metric: Literal["mean", "median"] = "mean",
    plot_estimator_ci: bool = False,
):
    """Plot the distribution of running-best over iterations for a single experiment. This
    will add a line with filled in confidence intervals around it to the axis.

    Args:
        objectives_per_experiment (np.ndarray): [num_experiments, num_steps] array with the
            objective observed for each of the steps.
        label (str): The label for this experiment
        ax (plt.Axes): Axis on which to plot
        linestyle (Literal[, optional): Linestyle to use in the plot
        confidence_interval (Optional[float], optional): Confidence interval (percentile) to plot
            with filled in regions. Defaults to plotting the standard deviation.
        metric (Literal[, optional): Which metric ot plot with the solid line: mean or median
        plot_estimator_ci (bool, optional): If True, plot the confidence intervals for the
            metric/estimator – i.e. plot the uncertainty in the mean/median rather than the
            overall running minimum. Defaults to False.
    """
    # running_best is an array of shape [num_experiments, num_steps]
    running_best = np.minimum.accumulate(objectives_per_experiment, axis=1)

    avg_metric = partial(np.mean, axis=-2) if metric == "mean" else partial(np.median, axis=-2)
    average = avg_metric(running_best)
    if confidence_interval is None:
        assert metric != "median", "Median ± std. is not a particularly meaningful representation of data."
        # Shade in +- standard deviation
        if plot_estimator_ci:
            std = bootstrap_standard_deviation(running_best, estimator=avg_metric)
        else:
            std = running_best.std(axis=0)
        lower = average - std
        upper = average + std
    else:
        if plot_estimator_ci:
            lower, upper = bootstrap_confidence_interval(running_best, estimator=avg_metric, ci=confidence_interval)
        else:
            # Shade in the confidence interval
            lower = np.percentile(running_best, 100 * (1 - confidence_interval) / 2, axis=0)
            upper = np.percentile(running_best, 100 * (1 + confidence_interval) / 2, axis=0)

    steps = np.arange(1, average.shape[0] + 1)
    ax.fill_between(steps, lower, upper, color=color, alpha=0.2)
    ax.plot(steps, average, color=color, alpha=0.9, label=label, linestyle=linestyle)


def plot_running_best_traces_dist_comparison(
    Y_runs_list: List[List[np.ndarray]],
    names: List[str],
    colors: List,
    ax: plt.Axes,
    batch_size: int,
    confidence_interval: Optional[float] = None,
    metric: Literal["mean", "median"] = "mean",
    plot_estimator_ci: bool = False,
):
    lines = ["-.", "--", "-", ":"]
    linecycler = cycle(lines)

    #  Plot the lines
    for objectives_per_experiment, name, color in zip(Y_runs_list, names, colors):
        _plot_running_best_trace_distribution(
            np.stack(objectives_per_experiment, axis=0),
            color,
            label=name,
            ax=ax,
            linestyle=next(linecycler),
            confidence_interval=confidence_interval,
            metric=metric,
            plot_estimator_ci=plot_estimator_ci,
        )

    # Get the total number of iterations:
    num_iterations = {len(Y_runs_list[i][0]) for i in range(len(Y_runs_list))}
    # Assert it's the same for each experiment

    if len(num_iterations) != 1:
        raise ValueError("The number of iterations should be the same")

    num_iterations = next(iter(num_iterations))  # Extract the only element in set

    _decorate_convergence_in_batches_axis(ax=ax, batch_size=batch_size, num_iterations=num_iterations)
    plt.legend()


def _plot_running_best_trace_samples(
    objectives_per_experiment: np.ndarray,  # Shape [num_experiments, num_steps]
    color,
    label: str,
    ax: plt.Axes,
    metric: Literal["mean", "median"] = "mean",
):
    # running_best is an array of shape [num_experiments, num_steps]
    running_best = np.minimum.accumulate(objectives_per_experiment, axis=1)
    num_runs = running_best.shape[0]
    average = running_best.mean(axis=0) if metric == "mean" else np.median(running_best, axis=0)

    steps = np.arange(1, average.shape[0] + 1)
    for run in range(num_runs):
        ax.plot(
            steps,
            running_best[run, :],
            alpha=min(0.2, 0.2 * 50 / num_runs),
            color=color,
            zorder=np.random.randint(1, 10 * num_runs),
        )
    ax.plot(steps, average, color=color, alpha=0.9, label=label, linewidth=2, zorder=num_runs * 100)


def plot_running_best_traces_samples_comparison(
    Y_runs_list: List[List[np.ndarray]],
    names: List[str],
    colors: List,
    ax: plt.Axes,
    batch_size: int,
    metric: Literal["mean", "median"] = "mean",
):
    #  Plot the lines
    for objectives_per_experiment, name, color in zip(Y_runs_list, names, colors):
        _plot_running_best_trace_samples(
            np.stack(objectives_per_experiment, axis=0), color, label=name, ax=ax, metric=metric
        )

    # Get the total number of iterations:
    num_iterations = {len(Y_runs_list[i][0]) for i in range(len(Y_runs_list))}
    # Assert it's the same for each experiment

    if len(num_iterations) != 1:
        raise ValueError("The number of iterations should be the same")
    num_iterations = next(iter(num_iterations))  # Extract the only element in set

    _decorate_convergence_in_batches_axis(ax=ax, batch_size=batch_size, num_iterations=num_iterations)
    plt.legend()


def plot_ranking_comparison(
    Y_runs_list: List[np.ndarray],
    names: List[str],
    colors: List,
    ax: plt.Axes,
    batch_size: int,
):
    y_runs_array = np.stack(Y_runs_list)  # shape [num_runs, num_repetitions, num_steps]
    y_cummin_runs = np.minimum.accumulate(y_runs_array, axis=-1)
    # Extract only y values when batch was collected
    y_cummin_per_batch_runs = y_cummin_runs[:, :, (batch_size - 1) :: batch_size]

    # Extract (fractional) ranking:
    runs_ranks = scipy.stats.rankdata(y_cummin_per_batch_runs, axis=0)  # same shape as y_runs_array

    #  Plot the rankings
    for experiment_ranks, name, color in zip(runs_ranks, names, colors):
        _plot_running_best_trace_distribution(experiment_ranks, color, label=name, ax=ax, metric="mean")

    # Get the total number of iterations:
    num_iterations = {len(Y_runs_list[i][0]) for i in range(len(Y_runs_list))}
    # Assert it's the same for each experiment

    if len(num_iterations) != 1:
        raise ValueError("The number of iterations should be the same")
    ax.set_xlim(1, runs_ranks.shape[-1])
    ax.set_ylabel("Rank at step")
    ax.set_xlabel("Num. batches seen")
    plt.legend()


def _decorate_convergence_in_batches_axis(ax: plt.Axes, batch_size: int, num_iterations: int):
    #  Draw subtle verticle lines indicating where a new batch was collected
    for i, x in enumerate(np.arange(batch_size, num_iterations, batch_size)):
        ax.axvline(x, color="gray", zorder=-2, linewidth=0.1)

    #  Add a new axis on top indicating the number of batches seen (compared to num. iterations)
    ax2 = ax.secondary_xaxis("top", functions=(lambda x: x / batch_size, lambda batch: batch * batch_size))
    num_batches = num_iterations // batch_size
    tick_delta = 1 + num_batches // 30  #  Ensures there are at most 30 ticks (over-crowded otherwise)
    ax2_new_ticks = np.arange(1, num_batches, tick_delta)
    ax2.set_xticks(ax2_new_ticks)
    ax2.set_xlabel("Num. batches seen")

    ax.set_xlim(1, num_iterations)
    ax.set_xlabel("Num. points collected")
    ax.set_ylabel("Running minimum")


def _plot_running_best_trace_againt_time(
    objectives_per_experiment: np.ndarray,  # Shape [num_experiments, num_steps]
    times_per_experiment: np.ndarray,  # [num_experiments, num_batches]
    batch_size: int,
    color,
    label: str,
    ax: plt.Axes,
    linestyle: Literal["-", "--", ":", "-."] = "-",
    confidence_interval: Optional[float] = None,
    plot_estimator_ci: bool = False,
    res: int = 500,
) -> plt.Axes:
    y_cummin_runs = np.minimum.accumulate(objectives_per_experiment, axis=-1)
    # Extract only y values when batch was collected
    y_cummin_per_batch_runs = y_cummin_runs[..., (batch_size - 1) :: batch_size]

    max_time = times_per_experiment.max(axis=-1).min()  # Take the max. time to be the max. time of the shortest run
    time_grid = np.linspace(0, max_time, res)

    running_best_per_time = []
    num_experiments = y_cummin_per_batch_runs.shape[0]
    for i in range(num_experiments):
        # Convert to interpolated values
        f = interp1d(times_per_experiment[i], y_cummin_per_batch_runs[i])
        running_best_per_time.append(f(time_grid))
    # Average across runs
    running_best_per_time = np.stack(running_best_per_time, axis=0)  # [num_experiments, num_batches]
    mean = running_best_per_time.mean(axis=0)
    if confidence_interval is None:  # Plot standard deviation
        if plot_estimator_ci:
            std = bootstrap_standard_deviation(running_best_per_time, estimator=lambda x: np.mean(x, axis=-2))
        else:
            std = running_best_per_time.std(axis=0)
        lower = mean - std
        upper = mean + std
    else:
        if plot_estimator_ci:
            lower, upper = bootstrap_confidence_interval(
                running_best_per_time, estimator=lambda x: np.mean(x, axis=-2), ci=confidence_interval
            )
        else:
            # Shade in the confidence interval
            lower = np.percentile(running_best_per_time, 100 * (1 - confidence_interval) / 2, axis=0)
            upper = np.percentile(running_best_per_time, 100 * (1 + confidence_interval) / 2, axis=0)

    ax.fill_between(time_grid, lower, upper, color=color, alpha=0.3)
    ax.plot(time_grid, mean, color=color, alpha=0.9, label=label, linestyle=linestyle)
    return ax


def plot_running_best_against_time_dist_comparison(
    Y_runs_list: List[List[np.ndarray]],
    times: List[List[np.ndarray]],
    names: List[str],
    colors: List,
    ax: plt.Axes,
    batch_size: int,
    confidence_interval: Optional[float] = None,
    plot_estimator_ci: bool = False,
):
    lines = ["-.", "--", "-", ":"]
    linecycler = cycle(lines)

    #  Plot the lines
    for times_per_experiment, objectives_per_experiment, name, color in zip(times, Y_runs_list, names, colors):
        _plot_running_best_trace_againt_time(
            times_per_experiment=np.stack(times_per_experiment, axis=0),
            objectives_per_experiment=np.stack(objectives_per_experiment, axis=0),
            color=color,
            label=name,
            ax=ax,
            batch_size=batch_size,
            linestyle=next(linecycler),
            confidence_interval=confidence_interval,
            plot_estimator_ci=plot_estimator_ci,
        )

    # Get the total number of iterations:
    num_iterations = {len(Y_runs_list[i][0]) for i in range(len(Y_runs_list))}
    # Assert it's the same for each experiment

    if len(num_iterations) != 1:
        raise ValueError("The number of iterations should be the same")

    num_iterations = next(iter(num_iterations))  # Extract the only element in set
    ax.set_xlabel("CPU time (s)")
    ax.set_ylabel("Running minimum")
    plt.legend()


def bootstrap_confidence_interval(
    x: np.ndarray,
    estimator: Callable[[np.ndarray], np.ndarray],
    ci: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    x_resampled = get_bootstrap_samples(x, num_samples=10000)
    y_resampled = estimator(x_resampled)
    lower = np.percentile(y_resampled, 100 * (1 - ci) / 2, axis=0)
    upper = np.percentile(y_resampled, 100 * (1 + ci) / 2, axis=0)
    return lower, upper


def bootstrap_standard_deviation(
    x: np.ndarray,
    estimator: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    x_resampled = get_bootstrap_samples(x, num_samples=10000)
    return estimator(x_resampled).std(axis=0)


def get_bootstrap_samples(x: np.ndarray, num_samples: int = 10000) -> np.ndarray:
    """
    If x has shape [N, ...], return an array x_resampled of shape [num_samples, N, ...] with each
    row x_resampled corresponding to one resampled version of x
    """
    resample_idxs = np.random.randint(0, x.shape[0], size=(num_samples, x.shape[0]))
    return x[resample_idxs]
