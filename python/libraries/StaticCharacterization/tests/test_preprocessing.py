# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import numpy as np
import pytest
from staticchar import preprocessing as pp


def test_background_estimate():
    data = np.concatenate([np.arange(10, 20), np.arange(5, 10)])
    assert pp.background_estimate(data, pp.BackgroundChoices.Minimum) == 5
    assert pp.background_estimate(data, pp.BackgroundChoices.FirstMeasurement) == 10
    with pytest.raises(ValueError):
        pp.background_estimate(data, None)  # type: ignore # auto
