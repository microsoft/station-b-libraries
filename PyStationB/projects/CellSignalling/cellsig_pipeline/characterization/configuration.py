# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Stores the data types for integration-based characterization.

Exports:
    Signal, represents a fluorescence signal
    IntegrationParameters, represents the interval over which we integrate
    CharacterizationConfig, the highest-level configuration class
"""
from typing import Dict

import pydantic


class Signal(pydantic.BaseModel):
    color: str


class IntegrationParameters(pydantic.BaseModel):
    """Represents an interval, over which we integrate."""

    t_min: float
    t_max: float


class CharacterizationConfig(pydantic.BaseModel):
    """

    Parameters:
        signals, a dictionary of signals recorder
        growth_signal, the absorbance signal from which we infer the cell density
        integration, the interval over which we integrate
    """

    signals: Dict[str, Signal]
    growth_signal: str
    integration: IntegrationParameters
