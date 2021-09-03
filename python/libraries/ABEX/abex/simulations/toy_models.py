# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Toy models: function which behaviour we understand, wrapped as Simulators.

For now it implements:
    - RadialFunction, a simple function of the form `f(x) = -(x-x0)**2`.
"""
from typing import Sequence, Tuple

import numpy as np
from abex.simulations.interfaces import SimulatorBase
from emukit.core import ContinuousParameter, ParameterSpace


class RadialFunction(SimulatorBase):
    def __init__(self, x0: Sequence[float], interval: Tuple[float, float] = (1, 1e4), log_space: bool = True):
        """A concave function with a unique global maximum:
            `f(X) = -(X-x0)**2`

        Args:
            x0: location of the maximum
            interval: open interval specifying the edge of the optimization hypercube
            log_space: if True, f is evaluated using log-space variables X'=log10 X and x0' = log10 x0. It does not
                affect the parameter space

        Properties:
            maximum (np.ndarray): location of the specified maximum. Shape (n,) where `n = len(x0)`.
        """
        # The location of prescribed maximum
        self.maximum: np.ndarray = np.array(x0).ravel()

        # Parameter space is a hypercube with edge `interval`
        self._parameter_space = ParameterSpace(
            [ContinuousParameter(f"x{i}", interval[0], interval[1]) for i, _ in enumerate(x0, 1)]
        )

        # Toggle the log-space flag
        self._log_space: bool = log_space

        self._validate_input()

    def _validate_input(self):
        """Check whether maximum is inside the parameter space."""
        X = self.maximum[None, :]
        if not np.all(self.parameter_space.check_points_in_domain(X)):
            raise ValueError("Specified maximum needs to be inside the defined parameter space.")

    @property
    def parameter_space(self) -> ParameterSpace:
        return self._parameter_space

    def _objective(self, X: np.ndarray) -> np.ndarray:
        """Calculates the objective for a batch of inputs."""
        # Broadcast `maximum` to the shape of X
        maximum_broadcast = np.vstack([self.maximum for _ in X])

        # If we work in the log space, we need to apply log10 to samples and the maximum
        if self._log_space:
            maximum_broadcast = np.log10(maximum_broadcast)
            X = np.log10(X)

        # Return the sum of -(X-maximum)**2 over all coordinates
        return -np.sum((X - maximum_broadcast) ** 2, axis=1)[:, None]  # type: ignore
