# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Visualisations of the fitted growth model.

Exports:
    plot_growth_model, used to plot the growth model
"""
from typing import Optional, Tuple
import numpy as np

import matplotlib.pyplot as plt

plt.switch_backend("Agg")
# flake8: noqa: E402
import staticchar.plotting.core as core
from staticchar.basic_types import ArrayLike, TIME_H
from staticchar.models import BaseModel
from staticchar.plotting.core import AnnotationSpec


def _visualise_model(
    ax: plt.Axes, model: BaseModel, ts: ArrayLike, model_color: str, growth_period_color: str, maturation_offset: float
) -> None:
    """An auxiliary function plotting the growth model `model` on axes `ax`."""
    # Visualise the fit
    ts = np.asarray(ts)
    ys = model.predict(ts)
    ax.plot(ts, ys, c=model_color, label="Predicted")

    # Visualise the time of maximal activity and the growth period
    core.mark_phase(
        ax,
        point=model.time_maximal_activity + maturation_offset,
        interval=model.growth_period + maturation_offset,
        color=growth_period_color,
    )


def plot_growth_model(
    ts: ArrayLike,
    ys: ArrayLike,
    model: Optional[BaseModel] = None,
    x_label: str = TIME_H,
    y_label: str = "Growth",
    x_lim: Tuple[float, float] = (0.0, 20.0),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    data_color: str = "C1",
    model_color: str = "C2",
    growth_period_color: str = "C3",
    maturation_offset: float = 0.0,
    annotation_spec: Optional[AnnotationSpec] = None,
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    """A high-level function used for visualisation of the growth model.

    Args:
        ts: time points
        ys: observations
        model: fitted growth model, optional
        x_label: label to be put on X axis
        y_label: label to be put on Y axis
        x_lim: limits for the X axis
        title: title for the plot
        ax: place to draw the model, optional
        data_color: color of the points corresponding to the observed data
        model_color: color of the fitted model line
        growth_period_color: if model is provided, the time of maximal activity and growth period are drawn
        maturation_offset: passed down to offset production region in plot
        annotation_spec: spec for whether to include labels and title

    Returns:
        - figure or None, None if axis was provided. If it was not provided, a new figure is produced.
        - axis
    """
    fig, core_ax = core.get_figure_and_axes(ax)

    # Plot the data
    core_ax.scatter(ts, ys, c=data_color, label="Observed")

    # Visualise the model
    if model is None:
        core_ax.text(
            0.1, 0.9, "Model Fitting Failed", color="red", fontweight="bold", transform=core_ax.transAxes
        )  # pragma: no cover
    else:
        _visualise_model(
            ax=core_ax,
            model=model,
            ts=ts,
            model_color=model_color,
            growth_period_color=growth_period_color,
            maturation_offset=maturation_offset,
        )

    if annotation_spec is not None:
        annotation_spec.apply(core_ax, xlabel=x_label, ylabel=y_label, title=title)

    # Set the limits
    core_ax.set_xlim(x_lim)

    return fig, core_ax
