# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This contains acquisition functions, ready to be optimized by point calculators and optimizers.

They should inherit from the classes implemented in ``gp.base`` and be probably implemented in either pytorch or JAX
backend.
"""
from gp.moment_matching.pytorch import (
    SequentialMomentMatchingEI,
    SequentialMomentMatchingLCB,
    SimultaneousMomentMatchingEI,
    SimultaneousMomentMatchingLCB,
)

__all__ = [
    # Sequential acquisitions
    "SequentialMomentMatchingEI",
    "SequentialMomentMatchingLCB",
    # Simultaneous acquisitions
    "SimultaneousMomentMatchingEI",
    "SimultaneousMomentMatchingLCB",
]
