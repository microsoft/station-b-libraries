# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Unit conversion capabilities. Used to normalize the inputs to the same unit (e.g. an input may be given in mM
for some samples and in uM for the other).

TODO: This module will become obsolete when pyBCKG starts supporting unit conversion.

Exports:
    change_units, the unit conversion function
"""
from typing import Dict, List

from pyBCKG.domain import Concentration


def _is_unit_allowed(unit: str):
    """Allows basic units, like "uM" and composite units, like "ng/ml"."""
    if unit.count("/") > 1:
        return False  # pragma: no cover

    for forbidden_character in "().,&*^%# ":
        if forbidden_character in unit:
            return False  # pragma: no cover
    return True


def conversion_factor_basic(unit1: str, unit2: str) -> int:
    """Finds the conversion factor from ``unit1`` to ``unit2``. Assumes that both units are basic
    (units of type "a/b" is *not* allowed).

    Raises:
        ValueError, if conversion if not possible

    Example:
        conversion_factor_basic("nM", "mM") = -6, as ``1 nM = 10**-6 mM``.
        conversion_factor_basic("kg", "g") = 3, as ``1 kg = 10**3 g``.
    """
    prefixes: Dict[str, int] = {
        "n": -9,
        "u": -6,
        "m": -3,
        "k": 3,
    }

    error = ValueError(f"Conversion from {unit1} to {unit2} is not possible.")

    # TODO: Make this more robust. E.g. m (mili) is not the same as M (mega).
    unit1, unit2 = unit1.lower(), unit2.lower()

    # In this case we can have "nM" and "uM", for example. We can however have "uA" and "uB" as well.
    if len(unit1) == len(unit2):
        # The case of trivial conversion.
        if unit1 == unit2:
            return 0
        # Otherwise, we check whether the suffix is right
        if unit1[1:] != unit2[1:]:
            raise error
        else:
            try:
                return prefixes[unit1[0]] - prefixes[unit2[0]]
            except KeyError:  # pragma: no cover
                raise error

    # In this case we can have conversion from "kM" to "M", but as well we can be given non-compatible "kA" and "B".
    elif len(unit1) > len(unit2):
        if unit1[1:] != unit2:
            raise error
        try:
            return prefixes[unit1[0]]
        except KeyError:  # pragma: no cover
            raise error

    else:  # In this case we can do a simple mathematical trick, swapping the units.
        return -conversion_factor_basic(unit2, unit1)


def change_units(concentration: Concentration, desired_unit: str) -> Concentration:
    """Changes units of the observed sample.

    Args:
        concentration: observation which unit we want to change
        desired_unit: the final unit to which the observation will be converted

    Returns:
        a new concentration, which unit matches the ``desired_unit``

    Raises:
        ValueError, if conversion is impossible
    """
    # Run basic data validation
    for unit in (concentration.units, desired_unit):
        if not _is_unit_allowed(unit):
            raise ValueError(f"Unit {unit} is not in the right format.")  # pragma: no cover

    # Now try to infer whether they are in the form "a/b" or just "a".
    basic_units1: List[str] = concentration.units.split("/")
    basic_units2: List[str] = desired_unit.split("/")

    if len(basic_units1) != len(basic_units2):
        raise ValueError(f"Conversion from unit {concentration.units} to {desired_unit} is not possible.")

    # Now we know that the unit conversion is possible.
    factor: int = conversion_factor_basic(basic_units1[0], basic_units2[0])

    if len(basic_units1) == 2:
        factor -= conversion_factor_basic(basic_units1[1], basic_units2[1])

    return Concentration(value=concentration.value * 10 ** factor, units=desired_unit)
