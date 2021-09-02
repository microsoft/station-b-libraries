# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This submodule implements useful helper functions for characterizing whole experiments (multiple wells).

Exports:
    Plate, a class for running characterization over a whole plate and storing results
"""

import collections
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Type, TypeVar, Union, cast

import matplotlib.pyplot as plt
from matplotlib.axes import SubplotBase
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from staticchar.basic_types import TIME, ArrayLike
from staticchar.config import CharacterizationConfig, CharacterizationMethod, GrowthModelType
from staticchar.datasets import Dataset
from staticchar.gradients import transcriptional_activity_ratio
from staticchar.integrals import integrate
from staticchar.models.base import BaseModel
from staticchar.models.gompertz import GompertzModel
from staticchar.models.logistic import LogisticModel
from staticchar.plotting.core import AnnotationSpec, mark_phase
from staticchar.plotting.growth_model import plot_growth_model
from staticchar.plotting.signals import plot_integration, plot_signals_against_reference, plot_signals_against_time
from staticchar.preprocessing import BackgroundChoices, subtract_background
from psbutils.arrayshapes import Shapes


# Base names (without the ".png") for plates output by characterize() for each plate.

#
SIGNALS_VS_TIME = "signals_vs_time"
SIGNALS_VS_REFERENCE = "signals_vs_reference"
GROWTH_MODEL = "growth_model"
INTEGRATION_FMT = "integration_{}"
CHARACTERIZATIONS_CSV = "characterizations.csv"
MODEL_VALUES_CSV = "model_values.csv"
VALUE_CORRELATIONS = "value_correlations"
RANK_CORRELATIONS = "rank_correlations"


def characterize_integral(
    subtracted: pd.DataFrame, config: CharacterizationConfig, reference_time: Optional[float] = None
) -> Dict[str, float]:
    """Run static characterization for a single well using the integral method.

    Args:
        subtracted: a data frame containing background-subtracted time-series observations
        config: a characterization config instance
        specified_interval: a specified TimePeriod that overrides what is in `config`
    """
    if reference_time is not None:
        config.time_window.reference = reference_time
    result: Dict[str, float] = {}
    interval = config.time_window + config.maturation_offset
    result.update(integrate(subtracted, config.signals, interval=interval))
    return result


def characterize_gradient(
    subtracted: pd.DataFrame, config: CharacterizationConfig, model: BaseModel, sample_id: str
) -> Dict[str, float]:
    """Run static characterization for a single well using the gradient method.

    Args:
        subtracted: a data frame containing background-subtracted time-series observations
        config: a characterization config instance
        model: a BaseModel (Logistic or Gompertz)
        sample_id: identifier for sample being characterized
    """
    gradient_result = transcriptional_activity_ratio(
        subtracted,
        config.signals,
        config.reference,
        config.signal_properties,
        model.parameters.growth_rate,
        model.growth_period,
        maturation_offset=config.maturation_offset,
        sample_id=sample_id,
    )
    result = {k: v.activity_value for k, v in gradient_result.items()}
    return result


def plot_timeseries(
    data: pd.DataFrame,
    color_dct: Dict[str, str],
    annotation_spec: AnnotationSpec,
    target_path: Optional[Path] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plots the provided data as a time series.

    Args:
      data: dataset to plot; must have a TIME ("time") column.
      color_dct: color to use for each dependent variable in data
      annotation_spec: specification of what annotations to put in the plot
      target_path: where to save the figure to, if present
      ax: Axis object to use; a new one is generated if absent.
      title: the title for the plot, if any

    Returns:
      the Axis object for the plot
    """

    def minus_mean_value(signal: str) -> float:
        return -cast(float, data[signal].mean())

    signals = sorted(set(data.columns).difference([TIME]), key=minus_mean_value)  # type: ignore
    if ax is None:  # pragma: no cover
        plt.figure(figsize=(6.4, 4.8))
        ax = cast(plt.Axes, plt.subplot())
    plot_signals_against_time(
        data, signals, ax, colors=color_dct, size=2.0, annotation_spec=annotation_spec, title=title
    )
    # sns.despine()
    if target_path is not None:  # pragma: no cover
        ax.get_figure().savefig(target_path)  # type: ignore
    return ax


def fit_model_for_well(
    config: CharacterizationConfig, subtracted_data: pd.DataFrame, well: Optional[str], sample_id: str
) -> Optional[BaseModel]:
    """
    Args:
      config: configuration to get some parameter values from
      subtracted_data: raw_data with background subtracted
      sample_id: used to identify the sample if there are problems.

    Returns:
       the fitted model, or None if fitting failed.
    """
    model_class = get_growth_model_class(config.growth_model)
    time_data = cast(ArrayLike, subtracted_data[TIME])
    signal_data = cast(ArrayLike, subtracted_data[config.growth_signal])
    try:
        model_params = model_class.fit(time_data, signal_data)  # type: ignore
    except RuntimeError:  # pragma: no cover
        logging.warning(f"Model fitting failed for well {well} (sample {sample_id})")
        return None
    return model_class(model_params)


def get_growth_model_class(gm_type: GrowthModelType) -> Type[BaseModel]:
    """
    Returns the growth model class (not instance) corresponding to the provided enum instance.
    """
    if gm_type == GrowthModelType.Logistic:
        return LogisticModel  # pragma: no cover
    if gm_type == GrowthModelType.Gompertz:
        return GompertzModel
    raise ValueError(f"Unexpected model type: {gm_type}")  # pragma: no cover


K = TypeVar("K")
S = TypeVar("S")
V = TypeVar("V")


def dictionary_with_subkey(subkey: S, dct: Dict[K, Dict[S, V]]) -> Dict[K, V]:
    return dict((key, val[subkey]) for (key, val) in dct.items() if subkey in val)


def plot_for_sample(
    config: CharacterizationConfig,
    subtracted_data: pd.DataFrame,
    model: Optional[BaseModel],
    ax_dict: Dict[str, plt.Axes],
    well_or_sample_id: str,
    annotation_spec: AnnotationSpec,
) -> None:
    """
    Fills in plots for the given sample.

    Args:
      config: configuration to get some parameter values from
      subtracted_data: raw data with background subtracted
      model: fitted model, whose parameters are used in the growth model plot; or None if fitting failed
      ax_dict: dictionary from plot names to Axis objects
      well_or_sample_id: to use in titles of plots
      annotation_spec: spec of what annotations to write
    """
    production_phase = config.time_window + config.maturation_offset
    try:
        production_phase.check_bounds_against(cast(Iterable[float], subtracted_data[TIME]), well_or_sample_id)
        phase_color = "green"
    except ValueError:  # pragma: no cover
        phase_color = "red"
    color_dct = {signal: config.signal_properties[signal].color for signal in config.signal_properties}
    relevant_signals = config.signals + [config.reference, config.growth_signal, TIME]
    relevant_data = cast(pd.DataFrame, subtracted_data[relevant_signals])
    # 8 characters should be enough to uniquely identify a sample without taking up too much space
    title = well_or_sample_id[:8]
    ax_timeseries = ax_dict.get(SIGNALS_VS_TIME, None)
    if ax_timeseries is not None:
        plot_timeseries(relevant_data, color_dct, ax=ax_timeseries, annotation_spec=annotation_spec, title=title)
        # Visualise the production phase
        mark_phase(ax_timeseries, interval=production_phase, color=phase_color, alpha=0.1)
    ax_ref = ax_dict.get(SIGNALS_VS_REFERENCE, None)
    if ax_ref is not None:
        plot_signals_against_reference(
            subtracted_data,
            signals=config.signals,
            reference=config.reference,
            colors=color_dct,
            ax=ax_ref,
            size=2.0,
            title=title,
            annotation_spec=annotation_spec,
        )
    ax_gm = ax_dict.get(GROWTH_MODEL, None)
    if ax_gm is not None:
        plot_growth_model(
            cast(ArrayLike, subtracted_data[TIME]),
            cast(ArrayLike, subtracted_data[config.growth_signal]),
            ax=ax_gm,
            model=model,
            growth_period_color=phase_color,
            annotation_spec=annotation_spec,
            title=title,
            maturation_offset=config.maturation_offset,
        )
    for signal in config.signals:
        ax_signal = ax_dict.get(INTEGRATION_FMT.format(signal), None)
        if ax_signal is not None:
            plot_integration(
                subtracted_data,
                signal,
                production_phase,
                ax_signal,
                annotation_spec=annotation_spec,
                title=title,
            )


def get_relative_ranks(data: Sequence[float]) -> np.ndarray:
    rank_dct1 = collections.defaultdict(list)
    for index, value in enumerate(sorted(data)):
        rank_dct1[value].append(index)
    rank_dct = dict((key, sum(lst) / len(lst)) for key, lst in rank_dct1.items())
    return np.array([rank_dct[value] for value in data]) / (len(data) - 1)


class Plate(Dict[str, Dict[str, float]]):
    """A class representing the characterization results for a given Dataset and CharacterizationConfig.

    Methods:
        __getitem__, so that the results can be accessed using ``dataset[key]`` syntax
        items, so that one can iterate over pairs (key, data frame) as in ``dict.items()``
    """

    def __init__(self, data: Dataset, config: CharacterizationConfig):
        """Initialize a Plate instance.

        Args:
            data: a dataset instance
            config: a characterization config instance
        """
        super().__init__()
        self.config = config
        self.data = data
        self.subtracted_columns = config.background_subtract_columns()
        self.subtract_strategy = BackgroundChoices.Minimum
        self.plots: Dict[str, Tuple[plt.Figure, np.ndarray]] = {}
        self.layout: Optional[Tuple[List[str], List[int]]] = None
        self.reference_time = self.get_reference_time()

    def get_reference_time(self) -> Optional[float]:
        """If possible, identify the average time of maximal growth in the reference wells."""
        if self.config.method != CharacterizationMethod.Integral or len(self.config.reference_wells) == 0:
            return None
        tmaxs = []
        for ref_id in self.config.reference_wells:
            subtracted = subtract_background(
                self.data[ref_id], columns=self.subtracted_columns, strategy=self.subtract_strategy
            )
            growth_model = fit_model_for_well(self.config, subtracted, None, ref_id)
            if growth_model is None:
                logging.warning(f"Model fitting failed on reference well {ref_id}")  # pragma: no cover
            else:
                tmaxs.append(growth_model.time_maximal_activity)
        if len(tmaxs) > 0:
            return float(np.mean(tmaxs))
        else:  # pragma: no cover
            logging.warning("Model fitting failed on all reference wells, so reference time not set.")
            return None

    def create_subplots(
        self,
        n_rows: int,
        n_cols: int,
        sharex: Union[bool, Literal["none", "all", "row", "col"]] = "col",
        sharey: Union[bool, Literal["none", "all", "row", "col"]] = "row",
        x_size: float = 2.5,
        y_size: float = 2.5,
        left_size: float = 0.8,
        right_size: float = 0.2,
        top_size: float = 0.3,
        bottom_size: float = 0.6,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Return a Figure and an n_rows x n_cols array of axes (SubplotBase objects). Each subplot is size
        x_size x y_size, and the figure has margins of the four indicated sizes. All "_size" arguments are
        in inches, and the margins are in addition to the space taken up by the axes array.
        """
        fig_x_size = x_size * n_cols + left_size + right_size
        fig_y_size = y_size * n_rows + top_size + bottom_size
        fig, ax = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_x_size, fig_y_size),
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
            gridspec_kw={
                "left": left_size / fig_x_size,
                "right": 1 - right_size / fig_x_size,
                "top": 1 - top_size / fig_y_size,
                "bottom": bottom_size / fig_y_size,
            },
        )
        ax_array = cast(np.ndarray, ax)
        Shapes(ax_array, f"{n_rows},{n_cols}")
        return fig, ax_array

    def set_up_figures(self) -> Tuple[int, int]:
        """
        Initializes self.plots, which is a dictionary from plot names to (figure, axes) pairs
        (which are distinct but initially identical). Plots are always created for SIGNALS_VS_TIME
        and GROWTH_MODEL. For the Gradient characterization method only, one is created for
        SIGNALS_VS_REFERENCE. For the Integral  method only, linear and log plots are
        also created for each signal. Linear plots have a separate vertical scale for each plot,
        allowing the area representing the integral to be seen clearly regardless of the range of the
        signal value, while log plots share a scale, allowing wells to be compared.
        """
        self.layout = self.data.plate_layout()
        if self.layout is None:
            n_rows, n_cols = self.decide_layout()
        else:  # pragma: no cover
            n_rows = len(self.layout[0])
            n_cols = len(self.layout[1])
        plot_names = [SIGNALS_VS_TIME, GROWTH_MODEL]
        if self.config.method == CharacterizationMethod.Integral:
            plot_names += [INTEGRATION_FMT.format(signal) for signal in self.config.signals]
        elif self.config.method == CharacterizationMethod.Gradient:
            plot_names += [SIGNALS_VS_REFERENCE]
        self.plots = {
            name: self.create_subplots(
                n_rows, n_cols, sharey=name.startswith(INTEGRATION_FMT.format("")) or name.startswith(SIGNALS_VS_TIME)
            )
            for name in plot_names
        }
        return n_rows, n_cols

    def characterize(self, output_dir: Path) -> None:
        """
        Run static characterization over the plate.
        """
        n_rows, n_cols = self.set_up_figures()

        growth_model_results: Dict[str, Dict[str, float]] = {}
        for index, (well, sample_id, frame) in enumerate(self.data.items_by_well()):
            growth_model = self.characterize_well(index, n_rows, n_cols, well, sample_id, frame)
            if growth_model is not None:
                growth_model_results[sample_id] = growth_model.to_dict()

        char_df = self.get_sorted_dataframe()
        char_df.index.name = "SampleID"
        char_df.to_csv(output_dir / CHARACTERIZATIONS_CSV)
        model_df = pd.DataFrame(growth_model_results).transpose()
        model_df.index.name = "SampleID"
        model_df.to_csv(output_dir / MODEL_VALUES_CSV)
        for name, (fig, _) in self.plots.items():
            fig.savefig(output_dir / f"{name}.png")
            plt.close(fig)
        self.plot_characterizations(growth_model_results, output_dir)

    def get_sorted_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame for the results, sorted by Well if present, otherwise by SampleID.
        """
        df = self.to_dataframe()
        if "Well" in df.columns:  # pragma: no cover

            def well_pair(well_series: pd.Series) -> pd.Series:
                return well_series.map(lambda well: (well[0], int(well[1:])))

            df = df.sort_values(by="Well", key=well_pair)  # type: ignore
        elif "SampleID" in df.columns:  # pragma: no cover
            df = df.sort_values(by="SampleID")
        return cast(pd.DataFrame, df)

    def plot_characterizations(self, growth_model_results: Dict[str, Dict[str, float]], output_dir: Path) -> None:
        """
        Creates a file correlations.png in output_dir for this plate. It shows a scatterplot for the ranks of
        the values values of each signal (including the growth signal) against each signal and condition, along
        with Spearman (rank) correlation coeffients and p values. We use ranks because the distributions can be
        highly non-normal in both the linear and log domains.

        Args:
            growth_model_results: passed in so we can plot the growth rate of the growth signal.
            output_dir: directory to write correlations.png to.
        """
        # Each row will contain scatterplots for one signal, with the growth signal first.
        signals = [self.config.growth_signal] + sorted(self.config.signals)
        # Each column will contain scatterplots for a signal or a condition
        conditions = self.get_condition_names()
        covariates = signals[1:] + conditions

        def get_color_and_dict(var_name: str) -> Tuple[str, Dict[str, float]]:
            """
            Returns the color to be used for the scatterplot if var_name is on the y axis, and a
            dictionary from sample IDs to values of var_name for that sample ID.
            """
            if var_name == self.config.growth_signal:
                return "black", dictionary_with_subkey("growth_rate", growth_model_results)
            if var_name in signals:
                return self.config.signal_properties[var_name].color, dictionary_with_subkey(var_name, self)
            # color will not be used for conditions, because it's chosen from the signal
            return "", dictionary_with_subkey(var_name, cast(Dict[str, Dict[str, float]], self.data.conditions))

        # We will print the (Spearman) correlation and its p-value above each plot. They will be in red if the
        # p-value is less than 0.05 after Bonferroni correction - which is a little too severe because Bonferroni
        # assumes the p-values are all independent, which they're not.
        #
        # With S signals and C conditions, there will be S*(S-1)/2 separate correlations between signals and
        # S*C between signals and conditions.
        n_correlations = len(signals) * ((len(signals) - 1) / 2 + len(conditions))
        significant_p_value = 0.05 / n_correlations

        for plot_type in [RANK_CORRELATIONS, VALUE_CORRELATIONS]:
            fig, ax = self.create_subplots(len(signals), len(covariates))
            for i, sig in enumerate(signals):
                ax[i, 0].set_ylabel(sig)
                color, sig_dct = get_color_and_dict(sig)
                for j, cov in enumerate(covariates[i:], i):  # noqa: E203
                    _, cov_dct = get_color_and_dict(cov)
                    # A valid sample is one that has a value for both sig and cov.
                    samples = sorted(set(sig_dct).intersection(cov_dct))
                    if len(samples) > 1:
                        cov_sig_pairs = [(cov_dct[sample], sig_dct[sample]) for sample in samples]
                        ax_ij = cast(SubplotBase, ax[i, j])
                        if plot_type == RANK_CORRELATIONS:
                            cov_data = get_relative_ranks([cov_value for (cov_value, _) in cov_sig_pairs])
                            sig_data = get_relative_ranks([sig_value for (_, sig_value) in cov_sig_pairs])
                            correlate_log_of_cov = False
                            correlate_log_of_sig = False
                        else:
                            cov_data = np.array([cov_value for (cov_value, _) in cov_sig_pairs])
                            sig_data = np.array([sig_value for (_, sig_value) in cov_sig_pairs])
                            xscale = self.get_correlation_scale(cov)
                            yscale = self.get_correlation_scale(sig)
                            ax_ij.set_xscale(xscale)  # type: ignore
                            ax_ij.set_yscale(yscale)  # type: ignore
                            correlate_log_of_cov = xscale == "log"
                            correlate_log_of_sig = yscale == "log"
                        ax_ij.scatter(cov_data, sig_data, s=4, color=color)  # type: ignore
                        if correlate_log_of_cov and cov_data.min() <= 0:  # pragma: no cover
                            # Omit negative points before taking logs for correlation
                            cov_data = cov_data[cov_data > 0]
                            sig_data = sig_data[cov_data > 0]
                        if correlate_log_of_sig and sig_data.min() <= 0:  # pragma: no cover
                            # Omit negative points before taking logs for correlation
                            cov_data = cov_data[sig_data > 0]
                            sig_data = sig_data[sig_data > 0]
                        r, pvalue = pearsonr(
                            np.log(cov_data) if correlate_log_of_cov else cov_data,
                            np.log(sig_data) if correlate_log_of_sig else sig_data,
                        )
                        title_color = "red" if pvalue <= significant_p_value else "black"
                        ax_ij.set_title(f"r={r:.3f}, p={pvalue:.4f}", fontsize=10, color=title_color)  # type: ignore
            for j, cov in enumerate(covariates):
                ax[len(signals) - 1, j].set_xlabel(cov)  # type: ignore
            fig.savefig(output_dir / f"{plot_type}.png")
            plt.close(fig)

    def get_correlation_scale(self, sig: str) -> str:
        """
        Scale for correlation plots is as specified in the config (default: "log") for signals,
        and "linear" for conditions.
        """
        properties = self.config.signal_properties
        if sig in properties:
            return properties[sig].correlation_scale
        if sig == self.config.growth_signal:
            return "log"
        return "linear"

    def get_condition_names(self) -> List[str]:
        """
        Returns the numerically-valued keys in the condition dictionary for an arbitrary sample in the plate.
        """
        dct = next(iter(self.data.conditions.values()))
        return sorted(key for key, val in dct.items() if isinstance(val, float))

    def characterize_well(
        self, index: int, n_rows: int, n_cols: int, well: Optional[str], sample_id: str, frame: pd.DataFrame
    ) -> Optional[BaseModel]:
        """
        Runs static characterization on a specific well (sample).

        Args:
          index: index of the plot within each figure; this affects how plots are annotated
          n_rows: number of rows in each plot, so we can work out from the index which plots are bottommost
          n_cols: number of columns in each plot, so we can work out from the index which plots are leftmost
          well: well ID, used to annotate plots when present
          sample_id: index for result of characterization; also used to annotate plots when "well" is None.
          frame: data to characterize.

        Returns:
          a fitted model when model fitting works, otherwise None.
        """
        if self.layout is None or well is None:
            row_index = index // n_cols
            col_index = index - n_cols * row_index
        else:  # pragma: no cover
            row_index = self.layout[0].index(well[0])
            col_index = self.layout[1].index(int(well[1:]))
        subtracted = subtract_background(frame, columns=self.subtracted_columns, strategy=self.subtract_strategy)
        model = fit_model_for_well(self.config, subtracted, well, sample_id)
        annotation_spec = AnnotationSpec(
            legend=(index == 0), title=True, xlabel=(row_index == n_rows - 1), ylabel=(col_index == 0)
        )
        ax_dict = {name: self.plots[name][1][row_index, col_index] for name in self.plots}  # type: ignore
        plot_for_sample(self.config, subtracted, model, ax_dict, well or sample_id, annotation_spec)
        well_result = {}
        try:
            if self.config.method == CharacterizationMethod.Integral:
                well_result = characterize_integral(subtracted, self.config, reference_time=self.reference_time)
            elif model is not None:
                well_result = characterize_gradient(subtracted, self.config, model, sample_id)
        except RuntimeError:  # pragma: no cover
            logging.warning(
                f"Unable to characterize well {well} (sample {sample_id})" f" using {self.config.method.value} method."
            )
        self[sample_id] = well_result
        return model

    def decide_layout(self):
        """
        Returns the number of rows and columns for a figure of len(self.data) plots. n_cols is usually 6,
        and n_rows is the number of rows needed to accommodate the required number of plots. However, if
        we can reduce n_cols without increasing n_rows, we do so. For example, if we need 8 plots, we
        can do it with two rows of 4 (not 6) plots each.
        """
        n_samples = len(self.data)
        n_cols = 6
        n_rows = 1 + (n_samples - 1) // n_cols
        while n_cols > 1 and 1 + (n_samples - 1) // (n_cols - 1) == n_rows:
            n_cols -= 1
        return n_rows, n_cols

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to dataframe, including the data conditions as additional columns"""
        conditions = pd.DataFrame.from_dict(self.data.conditions).transpose()
        plate_frame = pd.DataFrame.from_dict(self).transpose()
        return conditions.merge(plate_frame, left_index=True, right_index=True)
