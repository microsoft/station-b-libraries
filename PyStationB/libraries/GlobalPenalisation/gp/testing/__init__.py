# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Testing utilities."""
from gp.testing.naive import approximate_minimum
from gp.testing.gpy import evaluate_model, get_gpy_model, GPyModelWrapper


__all__ = [
    "approximate_minimum",
    "evaluate_model",
    "get_gpy_model",
    "GPyModelWrapper",
]
