# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""This is a very minimal implementation of integration-based characterization, supposed to replace the
deprecated `feature/characterization` branch of pyBCKG.

Exports:
    CharacterizationConfig, storing the information about the signals
    integral_characterization, a wrapper around integration using the trapezoidal rule
"""
from cellsig_pipeline.characterization.configuration import CharacterizationConfig  # noqa: F401
from cellsig_pipeline.characterization.methods import integral_characterization  # noqa: F401
