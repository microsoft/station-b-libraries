# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""The core plotting utilities."""
from dataclasses import dataclass, replace
from typing import Optional, Tuple
import matplotlib.pyplot as plt

plt.switch_backend("Agg")
# flake8: noqa: E402
from staticchar.basic_types import TimePeriod


def get_figure_and_axes(ax: Optional[plt.Axes]) -> Tuple[Optional[plt.Figure], plt.Axes]:
    """An auxiliary function, used to get a new figure is axis was not provided."""
    if ax is None:
        return plt.subplots()
    else:
        return None, ax


def mark_phase(
    ax: plt.Axes,
    point: Optional[float] = None,
    interval: Optional[TimePeriod] = None,
    color: str = "C3",
    alpha: float = 0.3,
) -> None:
    """Adds to `ax` a shadowed vertical region specified by `interval` and a vertical line at `point`.

    Args:
        ax: axes to which region and line should be applied
        point: X-axis location where a vertical line should be drawn. If None, no line is drawn
        interval: X-axis location where a vertical region should be drawn. If None, no region is drawn
        color: color of the line and the region
        alpha: alpha (controls opacity) of the region

    Note:
        This function is not pure.
    """
    if interval is not None:
        ax.axvspan(interval.tmin, interval.tmax, alpha=alpha, color=color)
    if point is not None:
        ax.axvline(point, color=color)


@dataclass
class AnnotationSpec:
    legend: bool = False
    title: bool = False
    xlabel: bool = False
    ylabel: bool = False

    def copy(self) -> "AnnotationSpec":  # pragma: no cover
        return replace(self)

    def apply(
        self, ax: plt.Axes, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None
    ):
        if self.legend:
            ax.legend()
        if self.title and title is not None:
            ax.set_title(title)
        if self.xlabel and xlabel is not None:
            ax.set_xlabel(xlabel)
        if self.ylabel and ylabel is not None:
            ax.set_ylabel(ylabel)
