# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
This module includes functions useful for computing the moment-matching minimum, which can be used
in many different array frameworks (currently torch and numpy).
"""
from gp.moment_matching.common.moment_matching_minimum import calc_cum_min_mean, calc_cum_min_var


__all__ = [
    "calc_cum_min_mean",
    "calc_cum_min_var",
]
