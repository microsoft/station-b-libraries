# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from staticchar.datasets import Dataset
from psbutils.misc import find_subrepo_directory


SUBREPO_DIR = find_subrepo_directory()


def test_loaddata() -> None:
    """Test the loading of data"""

    data_name = "SignificantDeath"
    data_folder = SUBREPO_DIR / "tests" / "test_data" / data_name
    data = Dataset(data_folder)
    assert len(data.conditions.values()) == 7
    assert len(data) == 7
    assert repr(data) == f"Dataset(path='{data_folder}')"
    assert all(data.get_a_frame(1) == sorted(data.items())[1][1])
