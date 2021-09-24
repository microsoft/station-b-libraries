# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""The JAX-powered implementation of the acquisition functions using the Clark's moment-matching approximation.

Note:
    All functions in ``clark.py`` and ``numeric.py`` should be JAX-compatible and functionally pure and can be
        differentiated freely.
    The submodule ``api.py`` provides the implementations of acquisitions functions in Emukit-compatible way.

Exports:
    SequentialMomentMatchingEI
    SequentialMomentMatchingUCB
    SimultaneousMomentMatchingLCB
    SimultaneousMomentMatchingEI
"""
from gp.moment_matching.jax.api import (
    SequentialMomentMatchingEI,
    SequentialMomentMatchingLCB,
    SimultaneousMomentMatchingLCB,
    SimultaneousMomentMatchingEI,
)

__all__ = [
    "SequentialMomentMatchingEI",
    "SequentialMomentMatchingLCB",
    "SimultaneousMomentMatchingLCB",
    "SimultaneousMomentMatchingEI",
]
