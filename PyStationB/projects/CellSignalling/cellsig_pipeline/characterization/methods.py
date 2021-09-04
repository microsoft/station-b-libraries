# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""The characterization methods.

Currently only the simples integration-based characterization is supported.

Exports:
    integral_characterization
"""
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _estimate_background(df: pd.DataFrame, signals: Iterable[str]) -> Dict[str, float]:  # pragma: no cover
    """Estimate the background value for each of the signals.

    Args:
        df, data frame with all timeseries. Its columns should contain `signals`
        signals, the signal names which minimum should be inferred

    Returns:
        dict, for each signal in `signals`, a float representing the background signal is given
    """
    return {signal: np.min(df[signal].values) for signal in signals}


def integral_characterization(
    df: pd.DataFrame, signals: Iterable[str], time_range: Tuple[float, float], time_column: str = "time"
) -> Dict[str, float]:  # pragma: no cover
    """The characterization method, via integration (with background subtraction).

    Args:
        df, a data frame with time series for each signal
        signals, columns of `df` representing signals to be characterized (integrated)
        time_range, a time interval over which all signals will be integrated
        time_column, a colum in `df` representing the time

    Returns:
        dict, for each signal in `signals`, a float representing the characterization value
    """
    background = _estimate_background(df=df, signals=signals)

    # Time vector
    time = df[time_column].values
    # Index
    time_min, time_max = time_range
    index = (time_min <= time) & (time <= time_max)  # type: ignore # auto

    results: Dict[str, float] = {}
    for signal in signals:
        x = time[index]
        y = df[signal].values[index]
        # Subtract background and integrate
        y_background: float = background[signal]

        results[signal] = np.trapz(y - y_background, x=x)  # type: ignore # auto

    return results
