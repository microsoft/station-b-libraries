# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBCKG.azurestorage.api as api
import pyBCKG.domain as domain
import seaborn as sns
from seaborn.palettes import _ColorPalette


def get_max(timeseriesmap, signal_guids: List[str]) -> float:
    maxval = -np.inf
    for signal_guid in signal_guids:
        for s in timeseriesmap:
            ts = timeseriesmap[s]
            if signal_guid in ts:
                maxval = max(maxval, max(ts[signal_guid]))
    return maxval


def get_condition_map(sample: domain.Sample, list_conditions: List[str]) -> Dict[str, domain.Condition]:
    """Returns a dictionary of the format condition_name: condition.
    The `list_conditions` can contain names or GUIDs of the conditions of interest."""
    name_to_condition_map: Dict[str, domain.Condition] = {c.reagent.name: c for c in sample.conditions}
    guid_to_condition_map: Dict[str, domain.Condition] = {c.reagent.guid: c for c in sample.conditions}
    condition_map: Dict[str, domain.Condition] = {}
    for condition_key in list_conditions:
        if condition_key in guid_to_condition_map:  # We have a GUID
            condition: domain.Condition = guid_to_condition_map[condition_key]
            name: str = condition.reagent.name
            condition_map[name] = condition
        elif condition_key in name_to_condition_map:  # We have a name
            condition_map[condition_key] = name_to_condition_map[condition_key]
        # else:
        #     print(f"WARNING: Reagent with guid or name {lc} not found in Sample {sample.guid}")
    return condition_map


def get_replicate_condition_str(
    sample: domain.Sample, list_conditions: List[str], reagent_comparison: List[str]
) -> str:
    condition_map: Dict[str, domain.Condition] = get_condition_map(sample, list_conditions)
    condition_str = ""
    for condition_name in sorted(condition_map):
        if condition_name not in reagent_comparison:
            condition_str += f"{condition_name}={condition_map[condition_name].concentration.to_str()}\n"
    for condition_name in sorted(condition_map):
        if condition_name in reagent_comparison:
            condition_str += f"reagent={condition_map[condition_name].concentration.to_str()}\n"
    return condition_str


def get_timestamp_string(observation: domain.Observation) -> str:
    if observation.observed_at is None:
        return "End of run"
    oa = datetime.strptime(observation.observed_at, "%Y-%m-%dT%H:%M:%S.%f0Z")  # type: ignore # auto
    return oa.strftime("%m/%d/%Y %H:%M:%S")


def has_reagent_condition(sample: domain.Sample, reagent_name: str) -> bool:
    condition_names = {condition.reagent.name for condition in sample.conditions}
    return reagent_name in condition_names


class Format(Enum):
    PNG = "png"
    PDF = "pdf"
    SVG = "svg"


class Scale(Enum):
    Linear = "linear"
    Log = "log"
    SymmetricalLog = "symlog"
    Logit = "logit"


@dataclass
class PlotSettings:
    x_label: str
    y_label: str
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    title: Optional[str] = None
    x_scale: Scale = Scale.Linear
    y_scale: Scale = Scale.Linear
    show_legend: bool = True
    dpi: int = 90
    separate_signals: bool = False
    label_map: Optional[domain.SIGNAL_MAP] = None
    list_conditions: List[str] = field(default_factory=list)

    def set_title(self, title: str) -> None:
        self.title = title

    def set_properties(self, ax: plt.Axes) -> None:
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)
        if self.title is not None:
            ax.set_title(self.title)
        ax.set_xscale(self.x_scale.value)
        ax.set_yscale(self.y_scale.value)

    @staticmethod
    def get_subplot_dimensions(n: int) -> Tuple[int, int]:
        if n == 1:
            return 1, 1
        elif n == 2:
            return 1, 2
        elif n == 3:
            return 1, 3
        elif n == 4:
            return 2, 2
        else:
            return math.ceil(n / 3), 3

    def get_condition_str(self, sample: domain.Sample) -> str:
        condition_map = get_condition_map(sample, self.list_conditions)
        reagent_names: List[str] = sorted(condition_map)
        condition_str = ""
        for name in reagent_names:
            condition_str += f"{name}={condition_map[name].concentration.to_str()} "
        return condition_str


ColorRGBA = Tuple[float, float, float, Optional[float]]
ColorMap = Dict[str, ColorRGBA]


class Plotting:
    def __init__(
        self,
        output_folder: Union[str, Path],
        conn: api.AzureConnection,
        color_palette: Union[str, _ColorPalette] = "tab20c",
    ):
        self.output_folder: Path = Path(output_folder)
        self.conn: api.AzureConnection = conn
        self.color_palette: _ColorPalette = (  # type: ignore # auto
            sns.color_palette(palette=color_palette) if isinstance(color_palette, str) else color_palette
        )

    def get_xy(self, ts: domain.TimeSeries) -> Tuple[List[float], List[float]]:
        ts_sorted = ts.data.sort_index()
        x = list(ts_sorted.index)
        y = list(ts_sorted.values)
        return x, y

    @staticmethod
    def get_color(
        signal: domain.Signal,
        color_map: ColorMap,
        default_color: ColorRGBA,
        label_map: Optional[domain.SIGNAL_MAP],
    ) -> ColorRGBA:
        signal_name: str = signal.to_label(label_map)
        signal_guid: str = signal.guid
        if signal_guid in color_map:
            return color_map[signal_guid]
        elif signal_name in color_map:
            return color_map[signal_name]
        return default_color

    def plot_timeseries(
        self,
        timeseries: List[domain.TimeSeries],
        settings: PlotSettings,
        _color_map: Optional[ColorMap] = None,
    ) -> plt.Figure:
        """Returns a plot of the `timeseries` data where the x-axis is time and y-axis is the signal measurement"""
        color_map: ColorMap = _color_map or {}

        if settings.separate_signals:
            n_signals = len(timeseries)
            n_rows, n_cols = settings.get_subplot_dimensions(n_signals)
            fig, axs = plt.subplots(
                nrows=n_rows,
                ncols=n_cols,
                figsize=(n_cols * 5, n_rows * 5),
                dpi=settings.dpi,
            )
            for i in range(n_cols - ((n_rows * n_cols) - n_signals), n_cols):
                fig.delaxes(axs[n_rows - 1, i])  # Removes plots from unused locations in subplots
            for ax, ts, palette in zip(axs.ravel(), timeseries, self.color_palette):
                color = self.get_color(ts.signal, color_map, palette, settings.label_map)
                x, y = self.get_xy(ts)
                settings.set_properties(ax)
                signal_name = ts.signal.to_label(settings.label_map)
                ax.plot(x, y, color=color, label=signal_name)
                ax.set_title(signal_name)
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)
            fig.tight_layout()
            if settings.title is not None:
                fig.suptitle(settings.title, y=1.05)
        else:
            fig, ax = plt.subplots(dpi=settings.dpi)
            settings.set_properties(ax)
            for palette_entry, ts in zip(self.color_palette, timeseries):
                color = self.get_color(ts.signal, color_map, palette_entry, settings.label_map)
                x, y = self.get_xy(ts)
                ax.plot(x, y, color=color, label=ts.signal.to_label(settings.label_map))
            if settings.show_legend:
                ax.legend(loc="best")
            fig.tight_layout()

        return fig

    def plot_observations(
        self,
        observations: domain.List[domain.Observation],
        settings: PlotSettings,
        _color_map: Optional[ColorMap] = None,
    ) -> plt.Figure:
        """Returns a plot of all `observations` grouped by the signal type."""
        color_map: ColorMap = _color_map or {}

        timestamps = []
        obs_map: Dict[str, Dict[str, List[domain.Observation]]] = {}  # A map from signal names to an observation map.
        for observation in observations:
            timestamp = get_timestamp_string(observation)
            timestamps.append(timestamp)
            sname = observation.signal.to_label(settings.label_map)
            if sname not in obs_map:
                obs_map[sname] = {}
            if timestamp not in obs_map[sname]:
                obs_map[sname][timestamp] = []  # A map of timestamp to measurement values.
            obs_map[sname][timestamp].append(observation)
        timestamp_list = list(set(timestamps))
        timestamp_list.sort()
        n_signals = len(obs_map.keys())
        n_rows, n_cols = settings.get_subplot_dimensions(n_signals)
        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(n_cols * 5, n_rows * 5),
            dpi=settings.dpi,
        )
        for i in range(n_cols - ((n_rows * n_cols) - n_signals), n_cols):
            fig.delaxes(axs[n_rows - 1, i])  # Removes plots from unused locations in subplots
        for ax, signal_name in zip(axs.ravel(), obs_map):
            point_map_x: Dict[ColorRGBA, List[int]] = {}
            point_map_y: Dict[ColorRGBA, List[float]] = {}
            for j, timestamp in enumerate(timestamp_list):
                if timestamp in obs_map[signal_name]:
                    for observation in obs_map[signal_name][timestamp]:
                        color = Plotting.get_color(
                            observation.signal,
                            color_map,
                            self.color_palette[i],  # type: ignore # auto
                            settings.label_map,
                        )
                        if color not in point_map_x:
                            point_map_x[color] = []
                            point_map_y[color] = []
                        point_map_x[color].append(j + 1)
                        point_map_y[color].append(observation.value)
            for c in point_map_x:
                x = point_map_x[c]
                y = point_map_y[c]
                ax.plot(x, y, "o", color=c)
            ax.set_title(signal_name)
            ax.set_ylim(bottom=0)
            sns.despline(top=True, right=True, ax=ax)  # type: ignore # auto
            xticks = range(0, len(timestamp_list) + 1)
            xlabels = [""] + timestamp_list
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, minor=False, rotation=40, ha="right")
        fig.tight_layout()
        return fig

    def plot_sample_data(
        self,
        sample: domain.Sample,
        settings: PlotSettings,
        _color_map: Optional[ColorMap] = None,
    ) -> plt.Figure:
        """Returns a plot of the timeseries data associated with the `sample`"""
        color_map: ColorMap = _color_map or {}

        file_ids: List[str] = self.conn.list_file_ids_of(sample.guid)
        data: StringIO = self.conn.get_timeseries_file(file_ids[0])
        df = pd.read_csv(data)
        signal_headers = list(df.columns)  # type: ignore # auto
        signal_headers.remove("time")
        tslist = []

        if len(settings.list_conditions) > 0:
            cstr = settings.get_condition_str(sample)
            if settings.title is None:
                settings.set_title(cstr)
            else:
                settings.set_title(f"{settings.title}\n{cstr}")

        for s_header in signal_headers:
            sig_df = df[["time", s_header]].dropna()  # type: ignore # auto
            ts_pairs = list(zip(sig_df["time"], sig_df[s_header]))  # type: ignore # auto
            signal = self.conn.get_signal_by_id(s_header)  # type: ignore # auto
            ts_data = {t: data for (t, data) in ts_pairs}
            ts = pd.Series(data=ts_data)
            tslist.append(domain.TimeSeries(ts, signal))

        return self.plot_timeseries(tslist, settings, color_map)

    def plot_replicates(
        self,
        experiments: List[domain.Experiment],
        settings: PlotSettings,
        reagent_comparison: List[str],
        signal_list: List[str],
    ) -> plt.Figure:
        col_headers = []
        for x in reagent_comparison:
            for y in signal_list:
                col_headers.append(x + " " + y)
        condition_list = []
        signal_map: Dict[str, Set[str]] = {s: set() for s in signal_list}
        for e in experiments:
            for sig in e.signals:
                sig_name = sig.to_label(settings.label_map)
                if sig_name in signal_map:
                    signal_map[sig_name].add(sig.guid)
            for sample in e.samples:
                condition_str = get_replicate_condition_str(sample, settings.list_conditions, reagent_comparison)
                if condition_str not in condition_list:
                    condition_list.append(condition_str)
        row_keys = sorted(condition_list)
        n_rows = len(row_keys)
        n_cols = len(col_headers)
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(n_cols * 5, n_rows * 5),
            subplot_kw={"xticks": [], "yticks": []},
        )
        for ax, col in zip(axes[0], col_headers):
            ax.set_title(col)

        for ax, row in zip(axes[:, 0], row_keys):
            ax.set_ylabel(row, labelpad=60, rotation=0)

        timeseriesmap = {}
        for expt in experiments:
            for sample in expt.samples:
                file_ids = self.conn.list_file_ids_of(sample.guid)
                if len(file_ids) > 0:
                    data: StringIO = self.conn.get_timeseries_file(file_ids[0])
                    timeseriesmap[sample.guid] = pd.read_csv(data)

        signal_limits = [get_max(timeseriesmap, list(signal_map[sig])) for sig in signal_list]
        n_signals = len(signal_list)
        for i, expt in enumerate(experiments):
            signal_map = {  # type: ignore # auto
                signal.get_plot_label(): signal.guid for signal in expt.signals  # type: ignore # auto
            }
            color = self.color_palette[i]
            samples = expt.samples
            rsamples_map = [
                {
                    get_replicate_condition_str(s, settings.list_conditions, reagent_comparison): s
                    for s in samples
                    if has_reagent_condition(s, r)
                }
                for r in reagent_comparison
            ]
            for j in range(len(row_keys)):
                rtimeseries = [timeseriesmap[rsample_map[row_keys[j]].guid] for rsample_map in rsamples_map]
                for k in range(n_signals):
                    ylim = signal_limits[k]
                    for z, rts in enumerate(rtimeseries):
                        ax = axes[j, k + (z * n_signals)]
                        x = rts["time"]
                        y = rts[signal_map[signal_list[k]]]
                        ax.plot(x, y, color=color)
                        ax.set_xlim(left=0)
                        ax.set_ylim([0, ylim])
        return fig

    def save_plot(
        self,
        fig: plt.Figure,
        file_name: str,
        dpi: int = 90,
        format: Format = Format.PNG,
    ) -> None:
        filename = f"{file_name}.{format.value}"
        plot_fp = self.output_folder / filename
        fig.savefig(plot_fp, dpi=dpi)
