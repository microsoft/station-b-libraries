# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This submodule implements useful pre-processing utilities.

Exports:
    subtract_background, a function to subtract background
    BackgroundChoices, various heuristics of estimating the background
"""
import enum
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from staticchar.basic_types import TIME, TimePeriod


class BackgroundChoices(enum.Enum):
    FirstMeasurement = "FirstMeasurement"
    Minimum = "Minimum"


def background_estimate(data: np.ndarray, strategy: BackgroundChoices) -> float:
    """A method for estimating the background signal."""
    if strategy == BackgroundChoices.FirstMeasurement:
        return data[0]
    elif strategy == BackgroundChoices.Minimum:
        return np.min(data)
    else:
        raise ValueError(f"Strategy {strategy} not recognized.")


def subtract_background(
    data: pd.DataFrame,
    columns: Iterable[str],
    blanks: Optional[Dict[str, float]] = None,
    strategy: BackgroundChoices = BackgroundChoices.FirstMeasurement,
) -> pd.DataFrame:
    """Returns a new data frame with background subtracted.

    Args:
        data: pandas data frame with measurements
        columns: columns for which
        blanks: background values to be subtracted
        strategy: for all signals in `columns` for which a blank measurement was not provided, we will use a heuristic
            to infer the background. Heuristics are implemented as BackgroundChoices.

    Returns:
        pandas data frame with the same columns as `data`. Columns that are not specified in `columns` argument
            are not modified

    Raises:
        KeyError, if `columns` is not a subset of `data.columns`

    Note:
        This is a *pure* function, meaning that original `data` is *not* modified.
        Also, if blanks are not provided, this subtract the *minimum*, what is not always the best choice
    """
    new_data = data.copy(deep=True)
    blanks_dct: Dict[str, float] = blanks or {}

    if not set(new_data.columns).issuperset(columns):  # pragma: no cover
        raise KeyError(f"Selected columns ({columns}) are not a subset of data frame columns ({new_data.columns}).")

    for col in columns:
        default_background: float = background_estimate(new_data[col].values, strategy=strategy)  # type: ignore # auto
        background: float = blanks_dct.get(col, default_background)
        new_data[col] = new_data[col] - background  # type: ignore

    return new_data


def select_time_interval(data: pd.DataFrame, interval: TimePeriod, time_column: str = TIME) -> pd.Series:
    """Returns a new data frame (or Series?), with rows such that data points are inside the interval.

    Args:
        data: data frame with column `time_column`
        interval: time interval to which the data is truncated
        time_column: the name of the column representing the time

    Returns:
        a new Series with rows which time are inside `interval`

    Note:
        This is a pure function.
    """
    ts = data[time_column].values
    index = interval.is_inside(ts)  # type: ignore # auto
    return data.iloc[index].reset_index(drop=True)
