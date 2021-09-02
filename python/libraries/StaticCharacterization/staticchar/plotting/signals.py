# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from staticchar.basic_types import ArrayLike, NPArrayLike, TimePeriod, TIME, TIME_H, NPArrayLike
from staticchar.plotting.core import AnnotationSpec
from typing import Dict, Iterable, Optional, Tuple, TypeVar, Union, cast
from matplotlib.markers import MarkerStyle

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt

plt.switch_backend("Agg")
# flake8: noqa: E402
import staticchar.plotting.core as core


T = TypeVar("T")


def _make_default(candidate: Union[Dict[str, T], T, None], keys: Iterable[str]) -> Dict[str, Optional[T]]:
    """Convert specified style to dictionary if it isn't already one.

    Args:
        candidate: a specified property (or dictionary of properties)
        keys: the keys needed
    """
    return candidate if isinstance(candidate, dict) else {key: candidate for key in keys}  # type: ignore


def _standardize_scatter_properties(
    signals: Iterable[str],
    colors: Union[Dict[str, str], str, None],
    markers: Union[Dict[str, str], str, None],
    size: Union[Dict[str, float], float, None],
) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]], Dict[str, Optional[float]]]:
    """Set properties for scatter plots, based on provided information.

    Args:
        signals: keys to distinguish different styles
        colors: marker color
        markers: marker style
        size: marker size
    """
    return _make_default(colors, keys=signals), _make_default(markers, keys=signals), _make_default(size, keys=signals)


def limit_range(data: NPArrayLike, factor: float = 0.0001) -> NPArrayLike:
    """
    Returns a variant on "data" in which low values are increased to be no less
    than factor times the maximum value. For use in log plots to avoid having a
    ridiculously large range.
    """
    min_allowed = np.max(data) * factor
    result = data.copy()
    result[data < min_allowed] = min_allowed
    return result


def plot_signals_against_time(
    data: pd.DataFrame,
    signals: Iterable[str],
    ax: Optional[plt.Axes] = None,
    time_column: str = TIME,
    colors: Union[Dict[str, str], str, None] = None,
    markers: Union[str, Dict[str, str], None] = None,
    size: Union[float, Dict[str, float], None] = None,
    title: Optional[str] = None,
    annotation_spec: Optional[AnnotationSpec] = None,
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    """Plots signals against time.

    Args:
        data: data frame
        signals: columns of `data` to be plotted
        ax: if provided, the plot will be plotted there
        time_column: column with time
        colors: a dictionary with color for each of signal present in `signals`. Alternatively, the default value
        markers: see `colors`
        size: see `colors`
        title: an optional title for the plot
        annotation_spec: instructions for what annotations to write
    """
    # Parse the dictionaries with default arguments
    colors2, markers2, size2 = _standardize_scatter_properties(signals, colors, markers, size)

    # Draw the scatter plots
    fig, ax = core.get_figure_and_axes(ax)

    def _get_for_signal(data):
        return data.get(signal) if isinstance(data, Dict) else data

    for signal in signals:
        ax.scatter(
            data[time_column],
            limit_range(cast(pd.Series, data[signal])),
            c=_get_for_signal(colors2),
            s=_get_for_signal(size2),
            marker=cast(Optional[MarkerStyle], _get_for_signal(markers2)),
            label=signal,
        )
    ax.set_yscale("log")

    if annotation_spec is not None:
        annotation_spec.apply(ax, xlabel=TIME_H, ylabel="Signals", title=title)

    return fig, ax


def plot_signals_against_reference(
    data: pd.DataFrame,
    signals: Iterable[str],
    reference: str,
    ax: Optional[plt.Axes] = None,
    fit_line: bool = True,
    colors: Union[Dict[str, str], str, None] = None,
    markers: Union[Dict[str, str], str, None] = None,
    size: Union[Dict[str, float], float, None] = None,
    linestyle: str = "--",
    linewidth: float = 1,
    alpha: float = 0.3,
    title: Optional[str] = None,
    annotation_spec: Optional[AnnotationSpec] = None,
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    """Plots `signals` against `reference`.

    Args:
        data: data frame with columns `signals` and `reference`
        signals: signals to be plotted on the Y-axis
        reference: signal to be plotted on the X-axis
        ax: if provided, axis on which the plots will be drawn
        fit_line: if the line of the best fit should be drawn
        colors: dictionary of colors for plotting different signals. Alternatively, this can be the default value
        markers: see `colors`
        size: see `colors`
        linestyle: line style of the fitted line (relevant only if `fit_line` is true)
        linewidth: see `linestyle`
        alpha: alpha (controls opacity) for the confidence bounds (relevant only if `fit_line` is true)
        title: an optional title for the plot
        annotation_spec: instructions for what annotations to write
    """
    colors2, markers2, size2 = _standardize_scatter_properties(signals, colors, markers, size)
    fig, ax = core.get_figure_and_axes(ax)

    for signal in signals:
        # Plot the points
        ax.scatter(
            data[reference],
            data[signal],
            label=signal,
            c=colors2[signal],
            s=size2[signal],
            marker=markers2[signal],  # type: ignore
        )

        # Plot a line
        if fit_line:
            x = np.linspace(np.min(data[reference]), np.max(data[reference]))
            slope, intercept, _, _, slope_error = stats.linregress(data[reference].values, data[signal].values)

            def line(s):
                return s * x + intercept

            ax.plot(x, line(slope), c=colors2[signal], label=signal, linestyle=linestyle, linewidth=linewidth)
            ax.fill_between(
                x, line(slope - slope_error), line(slope + slope_error), alpha=alpha, facecolor=colors2[signal]
            )

    if annotation_spec is not None:
        annotation_spec.apply(ax, xlabel=reference, ylabel="Signals", title=title)

    return fig, ax


def plot_integration(
    data: pd.DataFrame,
    signal: str,
    interval: Optional[TimePeriod],
    ax: Optional[plt.Axes] = None,
    time_column: str = TIME,
    fillcolor: Optional[str] = None,
    linestyle: Optional[str] = "k-",
    alpha: float = 0.3,
    title: Optional[str] = None,
    yscale: str = "linear",
    annotation_spec: Optional[AnnotationSpec] = None,
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    """Visualizes the integration characterization method, by showing the region of integration for a `signal`.

    Args:
        data: data frame with columns `signals` and `reference`
        signal: signal to be plotted on the Y-axis
        interval: X-axis location where the integration region should be drawn. If None, no region is drawn
        ax: if provided, axis on which the plots will be drawn
        fillcolor: color of integration region
        linestyle: line style of the signal plot
        linewidth: see `linestyle`
        alpha: alpha (controls opacity) for the integration region
        title: an optional title for the plot
        yscale: linear or log, for the y axis
        annotation_spec: instructions for what annotations to write
    """

    fig, ax = core.get_figure_and_axes(ax)

    t_data = np.array(data[time_column])
    s_data = np.array(data[signal])

    if interval is not None:
        locs = np.where(interval.is_inside(t_data))[0]  # type: ignore # auto
        ax.fill_between(t_data[locs], s_data[locs], color=fillcolor, alpha=alpha)

    ax.plot(t_data, s_data, linestyle, label="Signal")
    ax.set_yscale(yscale)

    if annotation_spec is not None:
        if title is None:  # pragma: no cover
            title = signal
        else:
            title = f"{title} {signal}"
        annotation_spec.apply(ax, title=title, xlabel=TIME_H)

    return fig, ax
