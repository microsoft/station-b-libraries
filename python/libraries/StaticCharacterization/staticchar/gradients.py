# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""A submodule implementing the gradient-based characterization."""
import dataclasses
import logging
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from staticchar.basic_types import TIME, ArrayLike, Reporter, TimePeriod


@dataclasses.dataclass
class TranscriptionalActivityRatio:
    """Store the results of applying gradient-based characterization"""

    activity_value: float
    activity_error: float
    gradient_value: Optional[float] = None
    gradient_error: Optional[float] = None


def _dy_dx(x: ArrayLike, y: ArrayLike) -> Tuple[float, float]:
    """Calculates the gradient of the least squares linear fit. For `y = ax+b`, it returns `a` and its error.

    Todo:
        1. Consider raising an explicit error if there are not enough values.
        2. Consider assessing the quality of the fit (e.g. `p`-value, `r**2`).
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, std_err


def transcriptional_activity_ratio(
    data: pd.DataFrame,
    signal_names: Iterable[str],
    reference_name: str,
    signal_properties: Dict[str, Reporter],
    growth_rate: float,
    growth_period: TimePeriod,
    maturation_offset: float = 0,
    time_column: str = TIME,
    sample_id: Optional[str] = None,
) -> Dict[str, TranscriptionalActivityRatio]:
    """Transcriptional activity ratio.

    *** Caution ***
        I am unsure if the growth rate we report in the growth model parameters (as in Gompertz or Logistic)
        is actually the right one. This is related to two separate reasons:
          - The growth rate is *probably* defined as `y'(t_max_activity)`, but I'm unsure if the models we use actually
            report this value.
          - Previously we fitted the growth curve to log(OD/OD(0)), meaning that the value of `mu` was very different.

    Args:
        data: data frame
        signal_names: columns in `data` representing the signals of interest
        reference_name: column in `data` representing the reference (e.g. the OD or mRFP1 signal)
        signal_properties: biochemical properties of the signal proteins
        growth_rate: growth rate at the time of maximal activity
        growth_period: growth period, so the plot `signal ~ reference` is linear
        maturation_offset: maturation time of the proteins
        time_column: time column, used to select the values in `growth_period`

    Note:
        Remember that the growth rate, maturation offset and chemical properties should have the same time units
        (here, we use hours throughout).

    Todo:
        The reported error on the transcriptional activity is underestimated -- the error on the growth rate is not
        taken under consideration.
    """
    ts: np.ndarray = data[time_column].values  # type: ignore # auto

    # Shift the growth period by maturation time
    interesting_period: TimePeriod = growth_period + maturation_offset

    # Select only the values of signal and reference that are inside the time period of interest
    index: np.ndarray = interesting_period.is_inside(ts)  # An array of boolean values
    # If no values, or only one, are inside the period of interest, then gradient calculation will fail.
    # As a fallback, we choose all values.
    if index.sum() < 2:  # pragma: no cover
        logging.warning(f"Using whole sequence as interval for gradient calculation for sample {sample_id}")
        index = np.full_like(index, True)
    reference = data[reference_name].values[index]

    # Loop over the signals, and calculate the activity ratios and errors
    def process_signal(signal_name: str) -> TranscriptionalActivityRatio:
        """Inner function to run method on a specified signal."""
        signal = data[signal_name].values[index]
        # Estimate the gradient of signal with respect to reference
        gradient, gradient_std_err = _dy_dx(x=reference, y=signal)  # type: ignore # auto

        # Ratiometric activity and its standard error
        rescale: float = signal_properties[signal_name].factor(growth_rate) / signal_properties[reference_name].factor(
            growth_rate
        )
        activity = gradient * rescale

        # TODO: This error is underestimated -- there is some error on growth rate as well...
        activity_std_err = gradient_std_err * rescale

        return TranscriptionalActivityRatio(
            gradient_value=gradient,
            gradient_error=gradient_std_err,
            activity_value=activity,
            activity_error=activity_std_err,
        )

    results = {signal_name: process_signal(signal_name) for signal_name in signal_names}

    return results
