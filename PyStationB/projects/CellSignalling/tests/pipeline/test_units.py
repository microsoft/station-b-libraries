# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from typing import List, Tuple

import pytest
from cellsig_pipeline.pybckg.units import Concentration, change_units

examples_no_slash: List[Tuple[Concentration, Concentration]] = [
    (Concentration(value=1, units="nM"), Concentration(value=0.001, units="uM")),
    (Concentration(value=1000, units="mM"), Concentration(value=1, units="M")),
    (Concentration(value=5, units="kg"), Concentration(value=5000, units="g")),
    (Concentration(value=5, units="mm"), Concentration(value=0.005, units="m")),
]


examples_with_slash: List[Tuple[Concentration, Concentration]] = [
    (Concentration(value=1, units="ng/ml"), Concentration(value=1e-9, units="g/ml")),
    (Concentration(value=1000, units="mM/l"), Concentration(value=1, units="M/l")),
    (Concentration(value=5, units="kg/l"), Concentration(value=5, units="g/ml")),
    (Concentration(value=1, units="g/l"), Concentration(value=1000, units="ug/ml")),
    (Concentration(value=10, units="nM/nM"), Concentration(value=10, units="M/M")),
]

valid_examples: List[Tuple[Concentration, Concentration]] = examples_no_slash + examples_with_slash

invalid_examples: List[Tuple[Concentration, Concentration]] = [
    (Concentration(value=1, units="g"), Concentration(value=1, units="nM")),
    (Concentration(value=1, units="ng"), Concentration(value=3, units="nA")),
    (Concentration(value=3, units="ng/ml"), Concentration(value=120, units="nM")),
    (Concentration(value=4, units="ng/ml"), Concentration(value=4, units="ng/a")),
]


@pytest.mark.parametrize("first,second", valid_examples)
def test_unit_conversion(first: Concentration, second: Concentration):
    assert change_units(first, second.units) == second
    assert change_units(second, first.units) == first


@pytest.mark.parametrize("first,second", valid_examples)
def test_unit_identity(first: Concentration, second: Concentration):
    assert change_units(first, first.units) == first
    assert change_units(second, second.units) == second


@pytest.mark.parametrize("first,second", invalid_examples)
def test_unit_raises_when_nonsense(first: Concentration, second: Concentration):
    with pytest.raises(ValueError):
        change_units(first, second.units)

    with pytest.raises(ValueError):
        change_units(second, first.units)
