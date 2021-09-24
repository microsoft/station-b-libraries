# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
WARNING: This sub-module is currently outdated.

The implementation needs to be changed from a correlation interface to covariance
interface in-line with the rest of the package.
"""
from gp.moment_matching.numpy.api import SequentialMomentMatchingEI

__all__ = [
    "SequentialMomentMatchingEI",
]
