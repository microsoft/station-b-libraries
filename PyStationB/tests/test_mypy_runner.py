# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import glob

import pytest

from scripts.mypy_runner import mypy_runner


@pytest.mark.timeout(10)
def test_mypy_runner():
    files = sorted(glob.glob("scripts/*.py"))
    mypy_runner(["-f"] + files)
