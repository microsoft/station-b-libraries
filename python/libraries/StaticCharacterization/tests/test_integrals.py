# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from pandas import DataFrame
from staticchar.basic_types import TIME, TimePeriod
from staticchar.integrals import integrate


def test_integrate():
    dct = {TIME: [0, 1, 2, 3], "gfp": [2.0, 3.0, 2.5, 2.0], "rfp": [1.0, 2.0, 1.5, 1.0]}
    df = DataFrame(dct)
    results = integrate(df, signals=["gfp", "rfp"])
    assert results["gfp"] == 7.5
    assert results["rfp"] == 4.5
    results2 = integrate(df, signals=["gfp", "rfp"], interval=TimePeriod(reference=1.5, left=0.5, right=0.5))
    assert results2["gfp"] == 2.75
    assert results2["rfp"] == 1.75
