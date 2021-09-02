# # Introduction
# In this notebook, we will load an example time series, fit a growth model
# and plot the signals.
#
# ## Load example time series
#
# Let's start by loading example time series data.
# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------

from typing import Iterable, List, Optional, cast

import matplotlib.pyplot as plt
import pytest
import seaborn as sns
import staticchar as ch
from psbutils.filecheck import Plottable, figure_found
from psbutils.misc import find_subrepo_directory
from staticchar.plotting.core import AnnotationSpec

SUBREPO_DIR = find_subrepo_directory()
S_SHAPE_FOLDER = SUBREPO_DIR / "tests/test_data/S-shape"


def plot_figure(name: str, ax: Optional[Plottable] = None) -> List[str]:
    sns.despine()
    found = figure_found(ax, f"test_introduction/{name}")
    plt.clf()
    return [] if found else [name]


@pytest.mark.timeout(10)
def test_introduction():
    dataset = ch.datasets.Dataset(S_SHAPE_FOLDER)  # type: ignore # auto

    raw_timeseries = dataset.get_a_frame()
    rth = raw_timeseries.head()

    # As we can see, there is some non-zero signal at the beginning, which we attribute to
    # the media absorbance and media fluorescence (as initially we have very low cell density).
    assert sorted(rth.keys().to_list()) == sorted([ch.TIME, "EYFP", "OD", "ECFP", "OD700", "mRFP1"])

    colors = {"EYFP": "yellow", "ECFP": "cyan", "mRFP1": "red", "OD": "black"}

    plt.figure(figsize=(6.4, 4.8))
    ax = cast(plt.Axes, plt.subplot())
    ch.plot_signals_against_time(raw_timeseries, signals=colors.keys(), time_column="time", ax=ax, colors=colors)
    ax.legend()
    figures_not_found = []
    figures_not_found += plot_figure("plot1_raw_timeseries", ax)

    # ## Pre-processing
    # Let's assume this is the background and subtract it.
    # (A more precise, but also costly alternative is to estimate this using several blanks).

    # In[ ]:

    subtracted = ch.subtract_background(
        raw_timeseries, columns=["OD", "ECFP", "EYFP", "mRFP1"], strategy=ch.BackgroundChoices.Minimum
    )
    ax = cast(plt.Axes, plt.subplot())
    ch.plot_signals_against_time(subtracted, signals=colors.keys(), time_column="time", ax=ax, colors=colors)
    ax.legend()
    figures_not_found += plot_figure("plot2_subtracted_timeseries", ax)

    # ## Run characterization on an example

    # In[ ]:

    yaml_path = find_subrepo_directory() / "tests/configs/integral_basic.yml"
    config = ch.config.load(yaml_path, ch.config.CharacterizationConfig)
    # config

    # ### Fitting a growth model
    #
    # Let's fit a growth model to the OD signal.

    model_params = ch.LogisticModel.fit(subtracted["time"], subtracted[config.growth_signal])  # type: ignore # auto
    model = ch.LogisticModel(model_params)

    # model_params = ch.GompertzModel.fit(subtracted["time"], subtracted[config.growth_signal])
    # model = ch.GompertzModel(model_params)

    print(f"Inferred parameters: {model_params}")
    print(f"Growth phase: {model.growth_period}")
    print(f"Time of maximal activity: {model.time_maximal_activity}")
    print(f"Inferred (log of) initial density: {model.initial_density(log=True)}")

    ch.plot_growth_model(subtracted["time"], subtracted[config.growth_signal], model=model)  # type: ignore # auto
    figures_not_found += plot_figure("plot3_growth_model_fit")

    # ### Plotting the data
    #
    # Some time after the growth phase, we should observe a similar exponential production
    # of the proteins. Suppose that this maturation time is about 50 minutes,
    # that is about 0.85 hours.
    #
    # Then, fluorescence signals should be linear when drawn with respect to each other.

    # Add offset to the growth phase
    production_phase = model.growth_period + config.maturation_offset

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # type: ignore
    ch.plot_signals_against_time(subtracted, signals=colors.keys(), time_column="time", ax=ax1, colors=colors)

    # Visualise the production phase
    ch.mark_phase(ax1, interval=production_phase, color="green", alpha=0.1)

    ch.plot_signals_against_reference(subtracted, signals=("EYFP", "ECFP"), reference="mRFP1", colors=colors, ax=ax2)

    figures_not_found += plot_figure("plot4_fluorescence_signals", f)

    # ### Truncate the time-series
    #
    # We see that this very well captures the growth phase of mRFP1 (the reference signal),
    # but is a bit too late for EYFP and ECFP -- we won't have a linear dependence between
    # the signals...
    #
    # Let's choose a more narrow interval.

    another_production_phase = ch.TimePeriod(reference=12, left=2, right=2)
    truncated_timeseries = ch.select_time_interval(subtracted, interval=another_production_phase)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # type: ignore
    ch.plot_signals_against_time(subtracted, signals=colors.keys(), time_column="time", ax=ax1, colors=colors)

    # Visualise the production phase
    ch.mark_phase(ax1, interval=another_production_phase, color="green", alpha=0.1)

    ch.plot_signals_against_reference(
        truncated_timeseries, signals=("EYFP", "ECFP"), reference="mRFP1", colors=colors, ax=ax2  # type: ignore # auto
    )
    figures_not_found += plot_figure("plot5_truncated")

    # Run method

    gradient, gradient_error = ch.transcriptional_activity_ratio(
        truncated_timeseries,  # type: ignore # auto
        config.signals,
        config.reference,
        config.signal_properties,
        model_params.growth_rate,
        model.growth_period,
        maturation_offset=config.maturation_offset,
    )
    # gradient

    # ### Integration-based characterization
    # Now assume that we want to integrate the signals over the production period.

    signals = ["EYFP", "ECFP"]
    ch.integrate(data=subtracted, signals=signals, interval=config.time_window)

    # Now plot the output

    f, axs = plt.subplots(1, len(config.signals), figsize=(12, 4))
    for signal, ax in zip(config.signals, cast(Iterable, axs)):
        ch.plot_integration(
            subtracted,
            signal,
            config.time_window,
            ax,
            fillcolor=colors[signal],
            annotation_spec=AnnotationSpec(title=True),
        )

    figures_not_found += plot_figure("plot6_integration", f)
    assert figures_not_found == [], f"Figures not found: {', '.join(figures_not_found)}"
