# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Useful data types.

Note:
    This module can be imported by an arbitrary staticchar submodule, meaning that we don't want to import *anything*
    from staticchar here (to prevent circular dependencies)."""

import abc
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd
import pydantic

ArrayLike = Union[List[float], np.ndarray, pd.Series]
NPArrayLike = Union[np.ndarray, pd.Series]

TIME = "time"
TIME_H = "Time (h)"


class TimePeriod(pydantic.BaseModel):
    """Represents a user-defined interval over which we apply static characterization."""

    reference: float  # Reference time-point
    left: float  # Duration before the reference time-point to consider
    right: float  # Duration after the reference time-point to consider

    @classmethod
    def from_minmax(cls, tmin: float, tmax: float):
        """
        Creates a TimePeriod from two bounds, with the reference at their midpoint.
        """
        assert tmin < tmax, "Invalid bounds for TimePeriod"
        reference = (tmin + tmax) / 2
        return TimePeriod(reference=reference, left=reference - tmin, right=tmax - reference)

    @pydantic.validator("right")
    def interval_must_be_positive(cls, v: float, values: Dict[str, float]) -> float:
        """
        Ensures that the left end of the time period is strictly before the right end.
        One value may be negative if the other is positive and larger; then the interval is
        valid although the reference time is outside it.
        """
        u = values.get("left", None)
        if u is not None and u + v <= 0:
            raise ValueError("Interval must have positive width")
        return v

    @property
    def tmin(self):
        """The minimum time of a TimePeriod"""
        return self.reference - self.left

    @property
    def tmax(self):
        """The maximum time of a TimePeriod"""
        return self.reference + self.right

    def is_inside(self, ts: ArrayLike) -> np.ndarray:
        """For a time series `ts` returns an array of booleans (True/False) saying which values are inside this
        interval."""
        ts = np.asarray(ts)
        return (self.tmin <= ts) & (ts <= self.tmax)

    def check_bounds_against(self, times: Iterable[float], sample_id: str) -> "TimePeriod":
        """
        If this TimePeriod does not fall entirely within the provided set of time values, raise a ValueError.
        """
        if self.tmin < min(times):
            raise ValueError(  # pragma: no cover
                f"TimePeriod.tmin = {self.tmin:.3f} is less than minimum datapoint time {min(times):3f}"
                f" for sample {sample_id}"
            )
        if self.tmax > max(times):
            raise ValueError(  # pragma: no cover
                f"TimePeriod.tmax = {self.tmax:.3f} is greater than maximum datapoint time {max(times):.3f}"
                f" for sample {sample_id}"
            )
        return self

    def __add__(self, value: float) -> "TimePeriod":
        """
        Returns a TimePeriod of the same size, shifted by "value".
        """
        return TimePeriod(reference=self.reference + value, left=self.left, right=self.right)


class AbstractReporter(abc.ABC):
    """A base class for any reporter, so that we can calculate the correction needed."""

    @abc.abstractmethod
    def factor(self, mu: float) -> float:  # pragma: no cover
        """This is the factor rescaling the gradient, needed to calculate the transcriptional activity.

        See eq. (5) of `B. Yordanov et al., A Computational Method for Automated Characterization of
        Genetic Components <https://doi.org/10.1021/sb400152n>`_

        For any growth signal (e.g. OD), this should be exactly 1.

        Args:
            mu: the growth rate

        Returns:
            the factor (the long fraction in eq. (5)) needed to calculate the transcriptional activity
        """
        raise NotImplementedError


class Reporter(pydantic.BaseModel, AbstractReporter):
    """
    [1] Elowitz, M., Leibler, S. A synthetic oscillatory network of transcriptional regulators.
        Nature 403, 335–338 (2000). https://doi.org/10.1038/35002125. See Box 1 on p. 337.
    [2] James Brown's PhD thesis: A design framework for self-organised Turing patterns in microbial populations (2011).
        See pp. 86 and 99.
    """

    maturation_rate: float  # Maturation rate in 1/h. These can be found in [2].
    degradation_rate: float = np.log(2) * 6  # Assume 10-minute half-life of the protein. See [1].
    translation_rate: float = 0.167 * 3600  # Convert the rate from seconds to hours. See [1].
    mRNA_degradation_rate: float = np.log(2) * 30  # Assume 2-minute half-life of mRNA. See [1].
    copy_number: float = 1.0  # We use the assumption of single copy number by default.
    color: str = "black"
    correlation_scale: str = "log"  # log or linear: transform to use for value correlation calculations.
    background_subtract: bool = False

    def factor(self, mu: float) -> float:
        return (
            self.mRNA_degradation_rate
            * (self.degradation_rate + mu)
            * (self.degradation_rate + self.maturation_rate + mu)
            / (self.maturation_rate * self.translation_rate * self.copy_number)
        )
