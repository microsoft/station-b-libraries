# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This module implements moment-matching approximation of different acquisition functions.

Structure:
  - ``acquisitions``: implemented acquisition functions
  - ``base``: the API for the acquisition functions, used by ``acquisitions`` and ``calculators``
  - ``calculators``: batch point calculators, used to suggest new batches to be collected
  - ``moment_matching``: implementations of Clark's moment matching algorithm
  - ``numeric``: NumPy-powered utility functions for common matrix operations
  - ``plotting``: plotting utilities
"""
import gp.calculators as calculators  # noqa: F401
import gp.moment_matching as mm  # noqa: F401
import gp.numeric as numeric  # noqa: F401
import gp.plotting as plotting  # noqa: F401
