# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""The three-parameter Gompertz growth model."""
import dataclasses
import warnings

import numpy as np
from scipy import optimize
from staticchar.basic_types import ArrayLike
from staticchar.models.base import BaseModel, CurveParameters


def _function(ts: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "overflow encountered")
        return a * np.exp(-b * np.exp(-c * ts))


@dataclasses.dataclass(frozen=True)
class _GompertzParameters:
    """The parameters of the curve
        y(t) = a * exp(-b * exp(-c * t)).

    They shouldn't probably be used widely (as they are hard to interpret),
    but they are very convenient mathematically.

    See also:
        CurveParameters, _reparametrize_to_model, _reparametrize_to_curve
    """

    a: float
    b: float
    c: float


def _reparametrize_to_model(params: CurveParameters) -> _GompertzParameters:
    """Auxiliary function for changing parametrization of the curve.

    See also:
        _reparametrize_to_curve
    """
    a: float = params.carrying_capacity
    c: float = params.growth_rate * np.e / a
    b: float = np.exp(c * params.lag_time + 1.0)

    return _GompertzParameters(a=a, b=b, c=c)


def _reparametrize_to_curve(params: _GompertzParameters) -> CurveParameters:
    """Auxiliary function for changing parametrization of the curve.

    See also:
        _reparametrize_to_model
    """
    return CurveParameters(
        carrying_capacity=params.a,
        growth_rate=params.a * params.c / np.e,
        lag_time=(np.log(params.b) - 1.0) / params.c,
    )


class GompertzModel(BaseModel):
    @property
    def time_maximal_activity(self) -> float:
        """The time of maximal growth rate"""
        params_model: _GompertzParameters = _reparametrize_to_model(self.parameters)
        return np.log(params_model.b) / params_model.c

    def predict(self, ts: ArrayLike) -> np.ndarray:
        """Gives the values of the model at timepoints `ts`."""
        ts = np.asarray(ts)
        model_params = _reparametrize_to_model(self.parameters)
        return _function(ts, a=model_params.a, b=model_params.b, c=model_params.c)

    @property
    def _log_initial_density(self) -> float:  # pragma: no cover
        """Return the natural logarithm of the initial density"""
        params_model: _GompertzParameters = _reparametrize_to_model(self.parameters)
        # We have `y(0) = a * exp(-b)`, so that `log(y(0)) = log(a) - b`
        return np.log(params_model.a) - params_model.b

    @staticmethod
    def _fit(ts: np.ndarray, ys: np.ndarray, initial_guess: CurveParameters, max_iterations: int) -> CurveParameters:
        """Finds optimal parameters of the growth model parameters."""
        initial_guess_model = _reparametrize_to_model(initial_guess)
        p0 = (initial_guess_model.a, initial_guess_model.b, initial_guess_model.c)

        estimates = optimize.curve_fit(_function, ts, ys, p0=p0, maxfev=max_iterations)[0]

        optimal_model = _GompertzParameters(a=estimates[0], b=estimates[1], c=estimates[2])

        return _reparametrize_to_curve(optimal_model)
