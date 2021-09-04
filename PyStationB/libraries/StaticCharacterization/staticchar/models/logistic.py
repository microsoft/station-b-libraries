# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""The three-parameter logistic growth model."""
import numpy as np
from scipy import optimize
from staticchar.basic_types import ArrayLike
from staticchar.models.base import BaseModel, CurveParameters


def _function(ts: np.ndarray, a: float, mu: float, lag: float) -> np.ndarray:
    """Solution of the logistic growth model"""
    inside_exp = 2.0 + 4 * (lag - ts) * mu / a
    return a / (1.0 + np.exp(inside_exp))


class LogisticModel(BaseModel):
    """The logistic model, as defined in `M. H. Zwietering et al. Modeling of the Bacterial Growth Curve
    <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC184525/>`_.
    """

    @property
    def time_maximal_activity(self) -> float:
        """The time of maximal growth rate"""
        params = self.parameters
        return params.lag_time + 0.5 * params.carrying_capacity / params.growth_rate

    @property
    def _log_initial_density(self) -> float:
        """Return the natural logarithm of the initial density"""
        params = self.parameters
        a, lag, mu = params.carrying_capacity, params.lag_time, params.growth_rate
        inside_exp = 2 + 4 * lag * mu / a
        return np.log(a) - np.log(1 + np.exp(inside_exp))

    def predict(self, ts: ArrayLike) -> np.ndarray:
        """Gives the values of the model at timepoints `ts`."""
        return _function(
            np.array(ts),
            a=self.parameters.carrying_capacity,
            mu=self.parameters.growth_rate,
            lag=self.parameters.lag_time,
        )

    @staticmethod
    def _fit(ts: np.ndarray, ys: np.ndarray, initial_guess: CurveParameters, max_iterations: int) -> CurveParameters:
        """Finds optimal parameters of the curve."""
        p0 = (
            initial_guess.carrying_capacity,
            initial_guess.growth_rate,
            initial_guess.lag_time,
        )
        estimates = optimize.curve_fit(_function, ts, ys, p0=p0, maxfev=max_iterations)[0]

        return CurveParameters(
            carrying_capacity=estimates[0],
            growth_rate=estimates[1],
            lag_time=estimates[2],
        )
