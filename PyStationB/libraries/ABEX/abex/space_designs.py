# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Various utilities for sampling the space (e.g. for initial batch data generation).

TODO: The approach in CellSignalling/scripts/initial_design.py needs to be coalesced with the approach
in data-loops. The initial_design.py script in CellSignalling should become core abex utility that is then called
by DataLoops, rather than having duplicate implementations of both.

Exports:
    DesignType, supported sampling designs (enum)
    suggest_samples, produces a batch of samples according to the selected design
"""
import itertools
from enum import Enum

import numpy as np
from emukit.core import ParameterSpace
from emukit.core.initial_designs.base import ModelFreeDesignBase
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.initial_designs.random_design import RandomDesign
from emukit.core.initial_designs.sobol_design import SobolDesign


class DesignType(Enum):
    """Supported design types."""

    GRID = "Grid"
    RANDOM = "Random"
    SOBOL = "Sobol"
    LATIN = "Latin"


class UniformGridDesign(ModelFreeDesignBase):
    """Generates a uniformly spaced mesh grid over a parameter space. Restricts the grid to the point_count
    specified by randomly dropping a subset of the points.
    """

    def get_samples(self, point_count: int) -> np.ndarray:  # pragma: no cover
        """Args:
            point_count, the number of samples to return

        Returns:
            numpy array of shape (point_count, n_dim), where n_dim is the length of the `parameter_space`
        """
        n_dim: int = len(self.parameter_space.get_bounds())
        approx_points_per_dim: float = point_count ** (1 / n_dim)
        points_per_dim: int = int(np.ceil(approx_points_per_dim))

        # Generate a grid for the whole space with n_dim**(points_per_dim) points
        one_dims = [np.linspace(low, upp, points_per_dim) for low, upp in self.parameter_space.get_bounds()]
        X = np.array(list(itertools.product(*one_dims)))

        # If there are too many points, remove a random subset.
        index = np.random.permutation(len(X))
        X = X[index, :]
        return X[:point_count, :]


def suggest_samples(parameter_space: ParameterSpace, design_type: DesignType, point_count: int) -> np.ndarray:
    """High-level function returning the samples accordingly to the specification provided.

    Attributes
        parameter_space: Emukit's parameter space
        design_type: method of generating an initial batch from parameter space. Supported design types are
            RANDOM, SOBOL, LATIN, and GRID
        point_count: number of points to be generated

    Returns
        numpy array of shape (point_count, n_dim), where n_dim is the length of the `parameter_space`
    """
    if design_type == DesignType.RANDOM:
        design = RandomDesign(parameter_space)
    elif design_type == DesignType.SOBOL:
        design = SobolDesign(parameter_space)  # pragma: no cover
    elif design_type == DesignType.LATIN:
        design = LatinDesign(parameter_space)
    elif design_type == DesignType.GRID:  # pragma: no cover
        design = UniformGridDesign(parameter_space)
    else:  # pragma: no cover
        raise ValueError(f"Design {design_type} not implemented.")
    return design.get_samples(point_count)
