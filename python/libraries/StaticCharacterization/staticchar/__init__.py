# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""StaticCharacterization -- time series characterization module.

The list of provided utilities, sorted into themes:

Preprocessing:
    select_time_interval, returns a new data frame, with rows that are in a specified time interval
    subtract_background, used to subtract background
    BackgroundChoices, implemented heuristics of inferring the background

Fitting growth models:
    CurveParameters, the three parameters used to parametrize e.g. the Gompertz or logistic curves
    GompertzModel, a three-parameter Gompertz model
    LogisticModel, a three-parameter logistic curve
    BaseModel, an abstract model so that more growth models can be implemented
    plot_growth_model, a utility for growth model visualisation

Plotting utilities:
    plot_growth_model, an already mentioned utility for model visualisation
    plot_signals_against_time, used to visualise many signals at once
    plot_signals_against_reference, plot one signal (or many) against another
    mark_phase, an auxiliary function allowing one to decorate a region

Characterization methods:
    integrate, numerical integration using the trapezoidal rule
    ReporterProperties, used to store the parameters to calculate transcriptional activity
    TimePeriod, a class representing a time interval
    TranscriptionalActivityRatio, the result of gradient-based characterization
    transcriptional_activity_ratio, the method of gradient-based characterization. (Caution: it needs to be adjusted.
        Read its docstring before using it and proceed with caution).

Helper methods:
    characterize_plate, applies a characterization method to all wells in a single experiment/plate

There is also a submodule used to load example data frames:
    datasets
"""
import staticchar.config as config  # noqa
import staticchar.models as models  # noqa
import staticchar.plotting as plotting  # noqa

from staticchar.basic_types import ArrayLike, Reporter, TimePeriod, TIME  # noqa
from staticchar.datasets import Dataset  # noqa
from staticchar.gradients import TranscriptionalActivityRatio, transcriptional_activity_ratio  # noqa
from staticchar.integrals import integrate  # noqa
from staticchar.models import BaseModel, CurveParameters, GompertzModel, LogisticModel  # noqa
from staticchar.plate import Plate  # noqa
from staticchar.plotting import (  # noqa
    plot_growth_model,
    plot_signals_against_reference,
    plot_signals_against_time,
    plot_integration,
    mark_phase,
)
from staticchar.preprocessing import select_time_interval, subtract_background, BackgroundChoices  # noqa
