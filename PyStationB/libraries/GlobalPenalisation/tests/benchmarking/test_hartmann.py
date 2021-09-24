# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from gp.benchmarking.benchmark_functions import Hartmann6
import scipy.optimize
import numpy as np
import pytest


def test_hartmann_minimum():
    x0 = np.zeros([6])
    res = scipy.optimize.minimize(Hartmann6.evaluate, x0=x0)
    found_minimum_loc = res["x"]
    expected_minimum_loc = np.asarray([0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054])
    found_minimum_val = res["fun"]
    expected_minimum_val = -3.32237
    assert pytest.approx(expected_minimum_loc, rel=1e-4, abs=0) == found_minimum_loc
    assert pytest.approx(expected_minimum_val, rel=1e-4, abs=0) == found_minimum_val
