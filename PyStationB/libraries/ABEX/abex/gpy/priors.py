# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Extensions of the priors specified in GPy"""
from typing import Sequence, Union

import GPy.core.parameterization.priors as priors
import numpy as np
from psbutils.arrayshapes import Shapes


class InverseGamma(priors.Gamma):  # pragma: no cover
    """
    Implementation of the inverse-Gamma probability function, coupled with random variables.

    This is a fix for the GPy.priors.InverseGamma implementation, which doesn't work since 2016:
    https://github.com/SheffieldML/GPy/issues/502

    :param a: shape parameter
    :param b: rate parameter (warning: it's the *inverse* of the scale)
    .. Note:: Bishop 2006 notation is used throughout the code
    """

    domain = priors._POSITIVE

    def __init__(self, a, b):
        self._a = float(a)
        self._b = float(b)
        self.constant = -priors.gammaln(self.a) + a * np.log(b)

    def __str__(self):
        """Return a string description of the prior."""
        return "iGa({:.2g}, {:.2g})".format(self.a, self.b)

    def lnpdf(self, x: np.ndarray) -> np.ndarray:
        """Return the log probability density function evaluated at x."""
        return Shapes(x, "X")(self.constant - (self.a + 1) * np.log(x) - self.b / x, "X")[-1]  # type: ignore # auto

    def lnpdf_grad(self, x: np.ndarray) -> np.ndarray:
        """Return the gradient of the log probability density function evaluated at x."""
        return Shapes(x, "X")(-(self.a + 1.0) / x + self.b / x ** 2, "X")[-1]  # type: ignore # auto

    def rvs(self, n: Union[int, Sequence[int], np.ndarray]) -> np.ndarray:
        """Return samples from this prior of shape n."""
        result = Shapes(1.0 / np.random.gamma(scale=1.0 / self.b, shape=self.a, size=n), f"{n}")[-1]
        return result  # type: ignore # auto
