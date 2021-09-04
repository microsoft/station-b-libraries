# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import abc
import dataclasses
from typing import Any, Dict, Optional

import numpy as np
from staticchar.basic_types import ArrayLike, TimePeriod


@dataclasses.dataclass
class CurveParametersContainer:
    """See `CurveParameters`. This class is a simple API to store e.g. probability distributions instead of
    point values."""

    carrying_capacity: Any
    growth_rate: Any
    lag_time: Any


@dataclasses.dataclass
class CurveParameters(CurveParametersContainer):
    """A parametrization of the S-shaped curve, which is easy to understand and assess visually.
    (In particular, it's easier to provide better starting values).

    These parameters are easy to understand, so that they should be used widely.

    Parameters:
        carrying_capacity: the upper asymptote,
        growth_rate: the slope at the time of maximal activity
        lag_time: the length of the "lower asymptote"
    """

    carrying_capacity: float
    growth_rate: float
    lag_time: float

    @staticmethod
    def guess_from_data(ts: ArrayLike, ys: ArrayLike) -> "CurveParameters":
        """Determine an appropriate initial guess for the CurveParameters

        Args:
            ts: Time-points of growth signal
            ys: Measurements of growth signal
        """
        # We will estimate the lag time basing on the total duration of the experiment
        total_time = np.max(ts) - np.min(ts)

        # Estimate the carrying capacity to be the maximal observed value (assumes any background has been subtracted)
        carrying_capacity = np.max(ys)

        # Denominator: Most time traces have roughly 3 phases: lag, exponential then saturation, so we approximate the
        # length of the exponential phase as total_time / 3, and use this as the denominator. The numerator is just
        # taken to be the maximum value - minimum value of the data, which is the carrying_capacity.
        growth_rate = carrying_capacity / (total_time / 3)

        return CurveParameters(
            lag_time=total_time / 2,  # Simple heuristic to take the mid-point of the time-window
            carrying_capacity=carrying_capacity,
            growth_rate=growth_rate,
        )


class BaseModel(abc.ABC):
    """An abstract growth model with three parameters, parametrized by CurveParameters.

    Properties:
        parameters, the curve parameters
        growth_period, the period of exponential growth
        time_maximal_activity, the time of maximal activity

    Methods:
        initial_density, estimates (the logarithm of) the initial density
        predict, predicts the values for given timepoints
        fit, infers the optimal parameters from observed data

    Abstract methods (need to be implemented by child classes):
        _log_initial_density
        predict
        time_maximal_activity
        _fit

    """

    def __init__(self, parameters: CurveParameters) -> None:
        """Initialize a growth model"""
        self._parameters: CurveParameters = parameters

    def __repr__(self) -> str:  # pragma: no cover
        """Display string for a growth model"""
        class_name = type(self).__name__
        return f"{class_name}(parameters={self.parameters})"

    def __str__(self) -> str:
        """Display string for a growth model"""
        return repr(self)

    def to_dict(self) -> Dict[str, float]:
        result = dataclasses.asdict(self.parameters)
        result["time_maximal_activity"] = self.time_maximal_activity
        result["log_initial_density"] = self.initial_density(log=True)
        return result

    @property
    def parameters(self) -> CurveParameters:
        """The parameters of a growth model"""
        return self._parameters

    @property
    @abc.abstractmethod
    def time_maximal_activity(self) -> float:
        raise NotImplementedError  # pragma: no cover

    @property
    @abc.abstractmethod
    def _log_initial_density(self) -> float:
        """Return the natural logarithm of the initial density"""
        raise NotImplementedError  # pragma: no cover

    def initial_density(self, log: bool = True) -> float:
        """Estimates (the natural logarithm of) the initial cell density.

        Args:
            log: if true, the natural logarithm of cell density will be provided instead of the raw value

        Returns:
            estimate of the value (or the logarithm of the value) of the initial cell density

        Note:
            if the initial cell density is expected to be small, it's better to estimate the logarithm of it
        """
        log_value = self._log_initial_density
        return log_value if log else np.exp(log_value)

    @property
    def growth_period(self) -> TimePeriod:
        """The exponential growth phase."""
        time_middle: float = self.time_maximal_activity
        lag: float = self.parameters.lag_time

        # The half-width is `time_middle - lag`
        return TimePeriod(reference=time_middle, left=time_middle - lag, right=time_middle - lag)

    @abc.abstractmethod
    def predict(self, ts: ArrayLike) -> np.ndarray:
        """Gives the values of the model at timepoints `ts`."""
        raise NotImplementedError  # pragma: no cover

    @staticmethod
    @abc.abstractmethod
    def _fit(ts: np.ndarray, ys: np.ndarray, initial_guess: CurveParameters, max_iterations: int) -> CurveParameters:
        """An actual backend for the `fit` method."""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def fit(
        cls, ts: ArrayLike, ys: ArrayLike, initial_guess: Optional[CurveParameters] = None, max_iterations: int = 2000
    ) -> CurveParameters:
        """Finds optimal parameters of the growth model parameters."""
        ts, ys = np.asarray(ts), np.asarray(ys)
        initial_guess = initial_guess or CurveParameters.guess_from_data(ts, ys)
        return cls._fit(ts=ts, ys=ys, initial_guess=initial_guess, max_iterations=max_iterations)
