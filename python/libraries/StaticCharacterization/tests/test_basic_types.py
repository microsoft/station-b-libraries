# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import numpy as np
import pytest
from staticchar.basic_types import TimePeriod


def test_time_period():
    tp1 = TimePeriod(reference=2.0, left=1.0, right=1.0)
    times = np.arange(0, 5)
    np.testing.assert_array_equal(tp1.is_inside(times), np.array([False, True, True, True, False]))
    tp2 = tp1 + 2.0
    assert tp2 == TimePeriod(reference=4.0, left=1.0, right=1.0)
    with pytest.raises(TypeError):
        tp1 + "foo"  # type: ignore # auto
    with pytest.raises(ValueError):
        TimePeriod(reference=2.0, left=1.0, right=-1.0)
    tp3 = TimePeriod.from_minmax(1.0, 3.0)
    assert tp3 == tp1
