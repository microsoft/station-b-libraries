# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Set-up for pytest tests. Sorts setting random seeds etc.
"""
import pytest
from jax.config import config


@pytest.fixture(autouse=True, scope="module")
def disable_jit():
    config.update("jax_disable_jit", True)
