# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from enum import Enum


class LocationIDs(Enum):
    """Enum to specify commonly used Sub-cellular locations."""

    ENDOPLASMIC_RETICULUM = "95"
    NUCLEUS = "191"
    CYTOPLASM = "86"


class TaxonomyIDs(Enum):
    """Enum to specify commonly used Taxonomies."""

    HUMAN = "9606"
    VSIV = "11285"
    JELLYFISH = "6100"
