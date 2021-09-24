# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Acquisition base classes.

This module provides base classes (a bare API) for two kinds of moment-matched approximation to acquisition functions.
(I.e., either all the points in the batch are optimized simultaneously or they are optimized sequentially).

These acquisition classes should be subclassed to create moment-matched approximations to q-EI, q-LCB, ...
Moreover, each backend (pytorch, JAX, ...) can subclass them.
"""
from gp.base.sequential import SequentialMomentMatchingBase, SequentialMomentMatchingBaseLCB
from gp.base.simultaneous import (
    SimultaneousMomentMatchingBase,
    SimultaneousMomentMatchingBaseLCB,
    MomentMatchingGMESBase,
)


__all__ = [
    "SequentialMomentMatchingBase",
    "SimultaneousMomentMatchingBase",
    # Classes implementing LCB (which require an additional beta parameter)
    "SequentialMomentMatchingBaseLCB",
    "SimultaneousMomentMatchingBaseLCB",
    # Generalised Max (min) value Entropy Search
    "MomentMatchingGMESBase",
]
