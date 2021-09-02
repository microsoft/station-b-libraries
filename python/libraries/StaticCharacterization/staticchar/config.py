# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""YAML configuration parser. Stores the data types for integration-based characterization.

Exports:
    CharacterizationConfig, the highest-level configuration class
"""
from enum import Enum
from os import PathLike
from typing import Callable, Dict, List, TypeVar

import pydantic
import yaml
from staticchar.basic_types import Reporter, TimePeriod

T = TypeVar("T")


class CharacterizationMethod(Enum):
    """The static characterization method to apply"""

    Gradient = "Gradient"
    Integral = "Integral"


class GrowthModelType(Enum):
    """A named growth model to fit cell growth data"""

    Gompertz = "Gompertz"
    Logistic = "Logistic"


class Query(pydantic.BaseModel):
    """A query for ratiometric characterization."""

    signal: str
    reference: str


class CharacterizationConfig(pydantic.BaseModel):
    """Configuration for static characterization.

    Parameters:
        growth_model: the model used to model cell growth
        growth_signal: the absorbance signal from which we infer the cell density
        maturation_offset: the adjustment for maturation time when identifying time of maximal cell growth
        method: the characterization method to apply (Integral or Gradient)
        reference: the signal used for ratiometric normalization (gradient only)
        reference_wells: the sample IDs of wells used to identify adjustments in the time window
        signal_properties: the parameters that define each fluorescent reporter
        signals: a list of signals to characterize
        time_window: the time window used for characterization

    Notes:
        When reference wells are specified, the `reference` property of the time_window is replaced by the average
        of the times of maximal growth rate of the reference wells.
    """

    time_window: TimePeriod = TimePeriod(reference=10.0, left=10.0, right=10.0)
    signals: List[str]
    reference: str
    growth_signal: str
    growth_model: GrowthModelType = GrowthModelType.Logistic
    maturation_offset: float = 0.85
    signal_properties: Dict[str, Reporter]
    method: CharacterizationMethod = CharacterizationMethod.Integral
    reference_wells: List[str] = []

    def colors(self) -> Dict[str, str]:
        """
        Returns a dictionary mapping each signal to its color in the config. Growth signal is always black.
        """
        cs = {k: rep.color for k, rep in self.signal_properties.items()}
        cs[self.growth_signal] = "black"
        return cs

    def background_subtract_columns(self) -> List[str]:
        """Identify which columns need to be background-subtracted."""
        cs = [self.growth_signal]  # Always background subtract the growth signal
        for k, rep in self.signal_properties.items():
            if rep.background_subtract:
                cs.append(k)
        return cs


def load(yaml_path: PathLike, class_factory: Callable[..., T]) -> T:
    """A generic method to load settings.

    Args:
        yaml_path: location of the YAML configuration file
        class_factory: a function generating pydantic objects storing configuration

    Returns:
        an object of type returned by class factory
    """
    with open(yaml_path) as file_handler:
        data_dict: dict = yaml.safe_load(file_handler)
        return class_factory(**data_dict)
