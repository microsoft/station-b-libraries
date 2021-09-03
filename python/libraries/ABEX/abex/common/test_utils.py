# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------

"""
This module is for code useful in more than one test file. We can't put it in the tests/ folder, because we
use an azureml library that creates a module called "tests" with an __init__.py file, which prevents us
importing anything from (our) "tests" module.
"""

import pandas as pd
from pandas.testing import assert_frame_equal


def assert_unordered_frame_equal(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> None:
    """
    Asserts that dataframes left and right are equal up to the ordering of the columns (columns can be in
    different order). Raises an AssertionError if they're not
    """
    assert_frame_equal(
        left.sort_index(axis=1), right.sort_index(axis=1), check_names=True, **kwargs  # type: ignore # check_names OK
    )
