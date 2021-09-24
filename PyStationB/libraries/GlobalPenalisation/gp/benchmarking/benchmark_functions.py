# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import enum
import numpy as np
from typing import Callable, Tuple
from functools import partial

from emukit.core import ParameterSpace, ContinuousParameter
from emukit.test_functions import branin_function, forrester_function, sixhumpcamel_function

from gp.numeric import sigmoid


class BenchmarkFunctions(enum.Enum):
    BRANIN = "branin"
    EGGHOLDER = "eggholder"
    SIMPLE_ADDITIVE = "simple-additive"
    HARD_ADDITIVE = "hard-additive"
    SIXHUMPCAMEL = "sixhumpcamel"
    FORRESTER = "forrester"
    HARTMANN6 = "hartmann6"
    COSINES_ADDITIVE = "cosines-additive"
    COSINES_SYMMETRIC = "cosines-symmetric"
    COSINES_ADDITIVE_SYMMETRIC = "cosines-additive-symmetric"

    def __str__(self) -> str:
        return self.value

    def toJSON(self) -> str:
        return self.value

    @classmethod
    def get_function_and_space(
        cls,
        function_name: str,
        dimensionality: int = 4,
    ) -> Tuple[Callable[[np.ndarray], np.ndarray], ParameterSpace]:
        if function_name.lower() == cls.SIMPLE_ADDITIVE.value:
            return additive_function()
        elif function_name.lower() == cls.HARD_ADDITIVE.value:
            return hard_additive_function()
        elif function_name.lower() == cls.BRANIN.value:
            return branin_function()
        elif function_name.lower() == cls.SIXHUMPCAMEL.value:
            return sixhumpcamel_function()
        elif function_name.lower() == cls.FORRESTER.value:
            return forrester_function()
        elif function_name.lower() == cls.HARTMANN6.value:
            return hartmann6_function()
        elif function_name.lower() == cls.EGGHOLDER.value:
            return eggholder_function()
        elif function_name.lower() in {
            cls.COSINES_ADDITIVE.value,
            cls.COSINES_ADDITIVE_SYMMETRIC.value,
            cls.COSINES_SYMMETRIC.value,
        }:
            # Above scenarios use the same target function, but construct a different model
            # (the model encodes different assumptions)
            return cosines_additive(dimensionality=dimensionality)
        else:
            raise ValueError(f"Not a valid test function: {function_name}")

    def __repr__(self) -> str:
        if self == self.SIMPLE_ADDITIVE:
            return "Simple additive function"


def additive_function() -> Tuple[Callable[[np.ndarray], np.ndarray], ParameterSpace]:
    """
    A simple 2D additive function with two large modes of similar magnitude, one slightly smaller than the other.

    The intended domain of the function is: x1 in [0, 10], x2 in [0, 10]
    """
    space = ParameterSpace([ContinuousParameter("x1", 0.0, 10.0), ContinuousParameter("x2", 0.0, 10.0)])
    return _additive_func, space


def hard_additive_function() -> Tuple[Callable[[np.ndarray], np.ndarray], ParameterSpace]:
    """
    A harder 2D additive function with many modes and many wiggles with relatively short lengthscale.

    The intended domain of the function is: x1 in [0, 10], x2 in [0, 10]
    """
    space = ParameterSpace([ContinuousParameter("x1", 0.0, 10.0), ContinuousParameter("x2", 0.0, 10.0)])
    return _additive_func, space


def gsobol_function(dim: int) -> Tuple[Callable[[np.ndarray], np.ndarray], ParameterSpace]:
    """
    gSobol test function used in "Batch Bayesian Optimization via Local Penalization" (Gonzalez et al. 2016)

    `f(x)=prod_{i=1}^{d} (|4 x_i-2| + a_1) / (1 + a_i)

    Just as in gonzalez et al. (2016), all the parameters a_i are set to 1.

    Args:
        dim (int):  The input dimensionality of the function.

    Returns:
        Test function and corresponding parameter space.
    """
    space = ParameterSpace([ContinuousParameter(f"x{i + 1}", -5.0, 5.0) for i in range(dim)])

    def gsobol_with_set_dim(x: np.ndarray) -> np.ndarray:
        return _gsobol_func(x=x, dim=dim)

    return gsobol_with_set_dim, space


def hartmann6_function() -> Tuple[Callable[[np.ndarray], np.ndarray], ParameterSpace]:
    """6-dimensional Hartmann Function.

    Domain: [0, 1]^6

    For reference, see: https://www.sfu.ca/~ssurjano/hart6.html
    """
    space = ParameterSpace([ContinuousParameter(f"x{i + 1}", 0.0, 1.0) for i in range(6)])
    return Hartmann6.evaluate, space


def eggholder_function() -> Tuple[Callable[[np.ndarray], np.ndarray], ParameterSpace]:
    """2-dimensional Eggholder function with *many* local minima

    Domain: [-512, 512]^2

    For reference, see: https://www.sfu.ca/~ssurjano/egg.html
    """
    space = ParameterSpace([ContinuousParameter(f"x{i + 1}", -512.0, 512.0) for i in range(2)])
    return _eggholder_function, space


def _eggholder_function(x: np.ndarray) -> np.ndarray:
    """Eggholder function (for reference, see: https://www.sfu.ca/~ssurjano/egg.html)"""

    x0 = x[..., 0]
    x1 = x[..., 1]

    term1 = -(x1 + 47) * np.sin(np.sqrt(np.abs(x1 + x0 / 2 + 47)))
    term2 = -x0 * np.sin(np.sqrt(np.abs(x0 - (x1 + 47))))

    y = term1 + term2
    return y[..., None]


def cosines_additive(dimensionality: int) -> Tuple[Callable[[np.ndarray], np.ndarray], ParameterSpace]:
    space = ParameterSpace([ContinuousParameter(f"x{i + 1}", -10.0, 10.0) for i in range(dimensionality)])
    return partial(_cosines_additive, dim=dimensionality), space


def additive_noise_wrapper(
    benchmark_function: Callable[[np.ndarray], np.ndarray],
    parameter_space: ParameterSpace,
    noise_std: float,
) -> Tuple[Callable[[np.ndarray], np.ndarray], ParameterSpace]:
    """
    Takes a tuple specifying a benchmark scenario (a function and parameter space) and adds noise to the function.
    """
    assert noise_std > 0, f"Noise standard deviation must be positive, but got {noise_std}."

    def noisy_benchmark_function(x: np.ndarray) -> np.ndarray:
        noise = np.random.normal(loc=0, scale=noise_std, size=[x.shape[0], 1])
        return benchmark_function(x) + noise

    return noisy_benchmark_function, parameter_space


def _additive_func(x: np.ndarray) -> np.ndarray:
    """
    A simple 2D additive function with two large modes of similar magnitude, one slightly smaller than the other.

    The intended domain of the function is: x1 in [0, 10], x2 in [0, 10]
    """
    #  Check points within intended domain
    assert np.all(x >= 0)
    assert np.all(x <= 10)

    assert x.ndim == 2
    assert x.shape[1] == 2

    x1, x2 = x.T

    def f1(x1):
        return -2 * sigmoid(1.2 * (x1 - 4)) - 0.03 * np.sin(2.3 * x1) + 0.8 * sigmoid(3.2 * (x1 - 9.5))

    def f2(x2):
        return (
            -np.sin(1.53 * x2 + 0.4) * 1.3
            - 0.015 * x2
            - 0.5 * sigmoid((x2 - 7) * 8)
            + 0.1 * np.exp(-((4.2 - x2) ** 2) / 5)
        )

    return (f1(x1) + f2(x2))[:, None]


def _additive_func_hard(x):
    """
    A harder 2D additive function with many modes and many wiggles with relatively short lengthscale.

    The intended domain of the function is: x1 in [0, 10], x2 in [0, 10]
    """
    #  Check points within intended domain
    assert np.all(x >= 0)
    assert np.all(x <= 10)

    assert x.ndim == 2
    assert x.shape[1] == 2

    x1, x2 = x.T

    def f1(z):
        term1 = 0.2 * ((-2 * sigmoid(1.2 * (z - 4)) - 0.03 * np.sin(2.3 * z) + 0.8 * sigmoid(3.2 * (z - 9.5))) + 3) * (
            np.sin(5 * z + 1) + 2.1
        ) + 0.1 * np.sin(11 * z)
        term2 = -1 * np.exp(-((6.2 - z) ** 2) / 0.1) - 0.58 * np.exp(-((3.2 - z) ** 2) / 0.1)
        return term1 + term2

    def f2(z):
        return (
            -0.5 * np.sin(1.53 * z + 0.4) * 1.3
            - 0.015 * z
            - 0.5 * sigmoid((z - 7) * 8)
            + 0.1 * np.exp(-((4.2 - z) ** 2) / 5)
            + np.cos(4 * z)
        ) * 0.6 + 0.08 * np.sin(8 * z + 0.5)

    return (f1(x1) + f2(x2))[:, None]


def _gsobol_func(x: np.ndarray, dim: int) -> np.ndarray:
    assert x.shape[-1] == dim, f"Input dimensionality {x.shape} doesn't match dim. of function {dim}"
    #  Each factor has the form (|4x - 2| + 1) / 2
    unreduced = (np.abs(4 * x - 2) + 1) / 2
    #  The function is a product of these factors
    reduced = unreduced.prod(axis=-1, keepdims=True)
    return reduced


def _cosines_func(x: np.ndarray) -> np.ndarray:
    assert x.shape[-1] == 2, f"Input dimensionality {x.shape} doesn't match dim. of function: 2"

    def g(z):
        return (1.6 * z - 0.5) ** 2

    def r(z):
        return 0.3 * np.cos(3 * np.pi * (1.6 * x - 0.5))

    terms = g(x) - r(x)
    return 1 - np.sum(terms, axis=-1, keepdims=True)


class Hartmann6:
    """Hartmann Function

    Global minimum at:

        [0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054]

    with a value of:
        -3.32236801141551

    with bounds [0, 1]^6.
    """

    A = np.asarray(
        [[10, 0.05, 3, 17], [3, 10, 3.5, 8], [17, 17, 1.7, 0.05], [3.5, 0.1, 10, 10], [1.7, 8, 17, 0.1], [8, 14, 8, 14]]
    )
    P = np.asarray(
        [
            [0.1312, 0.2329, 0.2348, 0.4047],
            [0.1696, 0.4135, 0.1451, 0.8828],
            [0.5569, 0.8307, 0.3522, 0.8732],
            [0.0124, 0.3736, 0.2883, 0.5743],
            [0.8283, 0.1004, 0.3047, 0.1091],
            [0.5886, 0.9991, 0.6650, 0.0381],
        ]
    )
    C = np.asarray([1, 1.2, 3, 3.2])

    @classmethod
    def evaluate(cls, x: np.ndarray) -> np.ndarray:
        #  Test (optimum test)
        inner_sum = cls.A.T * (x[..., None, :] - cls.P.T) ** 2
        outer_sum = cls.C * np.exp(-inner_sum.sum(axis=-1))
        summed = -outer_sum.sum(axis=-1)
        return summed[..., None]


def _cosines_additive(x: np.ndarray, dim: int) -> np.ndarray:
    """
    A symmetric, additive function on the domain [-10, 10]^d with a minimum at 0 of value 0.

    As such, it is well-modelled by an additive and/or symmetric kernel.
    """
    y = 1 - np.cos(x * 0.8) + (x / (10)) ** 2
    return y.sum(axis=-1, keepdims=True)


def _cosines_nonadditive(x: np.ndarray) -> np.ndarray:
    """A periodic function on the domain [-10, 10]^d with a minimum at 0 of value 0."""
    dist_to_center = np.sum(x ** 2, axis=-1, keepdims=True) ** 0.5
    y = 1 - np.cos(dist_to_center * 1.5) + (dist_to_center / 10) ** 2
    return y


def _cosines_changing_period_additive(x: np.ndarray, dim: int, overlap: int = 2) -> np.ndarray:
    """A periodic additive function on the domain [-10, 10]^d with a minimum at 0 of value 0."""
    y = 1 - np.cos(np.abs(x) ** 1.2) + (x / 10) ** 2
    return y.sum(axis=-1, keepdims=True)
