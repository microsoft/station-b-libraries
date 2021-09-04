# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from staticchar.basic_types import TimePeriod
from staticchar.models import LogisticModel
from staticchar.models.base import CurveParameters


def test_growth_period() -> None:
    model = LogisticModel(parameters=CurveParameters(carrying_capacity=1.0, growth_rate=1.0, lag_time=5.0))
    assert model.time_maximal_activity == 5.5
    assert model.growth_period == TimePeriod(reference=5.5, left=0.5, right=0.5)
    assert (
        model.__str__()
        == "LogisticModel(parameters=CurveParameters(carrying_capacity=1.0, growth_rate=1.0, lag_time=5.0))"
    )
