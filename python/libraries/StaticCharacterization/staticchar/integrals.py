# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Integration-based characterization methods."""
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from staticchar.basic_types import TIME, TimePeriod


def integrate(
    data: pd.DataFrame, signals: Iterable[str], interval: Optional[TimePeriod] = None, time_column: str = TIME
) -> Dict[str, float]:
    """Numerically integrates the signals (using the trapezoidal rule) over a specified interval.

    Args:
        data: data frame with signals and the time column
        signals: signals to be integrated. Must be a subset of `data` columns.
        interval: integration interval. If not provided, we integrate over the whole duration of the experiment
        time_column: column with respect to which the integration occurs. Must be a column in `data`.

    Returns:
        dict, keys are `signals` and values are the integrals

    Note:
        This is numerical integration. In particular, if the interval `[tmin, tmax]` doesn't contain enough data points,
        the calculated value may be quite far from the ideal result.
    """
    # If interval not provided, take everything
    interval = interval or TimePeriod(reference=0.0, left=np.inf, right=np.inf)

    # Time vector
    time = data[time_column].values

    # We want to take only the values that are inside the specified time interval
    index = interval.is_inside(time)  # type: ignore # auto

    results: Dict[str, float] = {}
    for signal in signals:
        x = time[index]
        y = data[signal].values[index]
        results[signal] = np.trapz(y, x=x)

    return results
