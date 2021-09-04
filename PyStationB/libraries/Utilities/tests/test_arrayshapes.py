# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from typing import List

import numpy as np
import pandas as pd
import pytest
from psbutils.arrayshapes import CHAIN_LENGTH_KEY, Bindings, ShapeInferrer, Shapes

A2 = np.zeros((2,), dtype=np.int32)
A24 = np.zeros((2, 4), dtype=np.float64)
A2468 = np.zeros((2, 4, 6, 8))
A146 = np.zeros((1, 4, 6))


def test_shapes_basic():
    Shapes(A2, "2")  # OK
    Shapes()(A2, "2")  # initialize empty and add __call__
    Shapes(["foo", "bar"], "2")  # OK; list can be coerced to array
    Shapes([["foo", "bar"]], "1, 2")  # OK; list can be coerced to array, spaces ignored
    with pytest.raises(ValueError):
        Shapes(A2, "3,")  # 2 != 3; final comma ignored


def test_shapes2():
    Shapes(A2, "X")  # trivially OK: X is bound to 2
    assert Shapes(A24, "X,Y").bindings == {"X": 2, "Y": 4}  # trivially OK: X is bound to 2, Y is bound to 4
    with pytest.raises(ValueError):
        Shapes(A24, "X,X")  # X is bound to 2, so cannot be re-bound to 4
    Shapes(A24, ("X, 2*X"))  # OK: X is bound to 2, 2*X evaluates to 4
    Shapes(A24, "X, (X+6)/2")  # OK: X is bound to 2, (2+6)/2 = 4.
    Shapes(A24, "X, (X+6)/2")  # OK: X is bound to 2, (2+6)/2 = 4.


def test_shapes_bad_syntax():
    with pytest.raises(ValueError):
        Shapes(A24, "X, 2*Y")  # Y is part of an expression but has not been bound yet
    with pytest.raises(ValueError):
        Shapes(A24, "X,Foo*2")  # variables in expressions have to be single characters, not splice variables
    with pytest.raises(ValueError):
        Shapes(A2, "X|Y")  # "|" is not a valid operator
    with pytest.raises(ValueError):
        Shapes(A24, "X, (X+6/2")  # missing closing parenthesis
    with pytest.raises(ValueError):
        Shapes(A24, "X,X/0")  # division by zero not allowed


def test_shapes4():
    with pytest.raises(ValueError):
        Shapes(A2, "X")(A24, "Y,X")  # bindings apply across different arrays
    Shapes(A2, "X")(A24, "Y,X*2")  # OK; X=2, so X*2=4 is OK in second array
    with pytest.raises(ValueError):
        Shapes(None, "X")  # cannot have shape without array
    with pytest.raises(ValueError):
        Shapes(A2)  # cannot have array without shape


def test_shapes5():
    Shapes(A24, "X, 2*X", [np.int32, np.float64])
    with pytest.raises(TypeError):
        Shapes(A24, "X, 2*X", np.int32)


def test_shapes_splice_variables():
    assert Shapes(A2468, "XX,Y,Z").bindings == {"XX": (2, 4), "Y": 6, "Z": 8}
    assert Shapes(A2468, "X,YY,Z").bindings == {"X": 2, "YY": (4, 6), "Z": 8}
    assert Shapes(A2468, "X,Y,ZZ").bindings == {"X": 2, "Y": 4, "ZZ": (6, 8)}
    assert Shapes(A24, "X,YY,Z").bindings == {"X": 2, "YY": (), "Z": 4}
    with pytest.raises(ValueError):
        Shapes(A2468, "XX, YY, Z")  # multiple splice variables
    with pytest.raises(ValueError):
        Shapes(A2, "X, YY, Z")  # too many variables even if YY is empty
    Shapes(A2468, "X,YY,Z")(A146, "1,YY")  # YY expands to 4,6 in both expressions


def test_shapes_where():
    Shapes(A2, "X")(A24, "X,Y").where(lambda bdgs: bdgs["Y"] == 2 * bdgs["X"])
    with pytest.raises(ValueError):
        Shapes(A2, "X")(A24, "X,Y").where(lambda bdgs: bdgs["Y"] == 3 * bdgs["X"])


def test_shapes_infer():
    Shapes.clear()
    for n in range(1, 5):
        Shapes(np.concatenate([A2] * n), "X")(np.concatenate([A24] * n), "Y,Z")
    lines = Shapes.infer()
    assert len(lines) == 3  # lines[0] is header line
    assert lines[1].endswith(": Y=X")
    assert lines[2].endswith(": Z=4")


def infer_for_bindings(blist: List[Bindings]) -> List[str]:
    return ShapeInferrer(blist).constraints()


def test_shapes_infer_for_bindings():
    blist: List[Bindings] = []
    assert infer_for_bindings(blist) == []
    blist = [{"X": 1, "Y": 1}, {"X": 2, "Y": 2}, {"X": 5, "Y": 5}]
    assert infer_for_bindings(blist) == ["Y=X"]
    blist = [{"X": 1, "Y": 1}, {"X": 2, "Y": 1}]
    assert infer_for_bindings(blist) == ["Y=1"]
    blist = [{"X": 1, "Y": 1}]
    assert infer_for_bindings(blist) == []  # don't infer anything from a single instance
    blist = [{"X": 1, "Y": 1}, {"X": 2, "Y": 3}]
    assert infer_for_bindings(blist) == ["Y=2*X-1"]
    blist = [
        {"X": 1, "Y": 2},
        {"X": 2, "Y": 4},
        {"X": 3, "Y": 6},
    ]
    assert infer_for_bindings(blist) == ["Y=2*X"]
    blist = [
        {"X": 1, "Y": 2},
        {"X": 2, "Y": 5},
        {"X": 3, "Y": 6},
    ]
    assert infer_for_bindings(blist) == []  # because not linear
    blist = [
        {"X": 1, "Y": 2, "Z": 2},
        {"X": 2, "Y": 4, "Z": 3},
        {"X": 3, "Y": 6, "Z": 4},
    ]
    assert infer_for_bindings(blist) == ["Y=2*X", "Z=X+1"]
    blist = [
        {"X": 2, "Y": 3, "Z": 6},
        {"X": 2, "Y": 4, "Z": 8},
        {"X": 3, "Y": 5, "Z": 15},
        {"X": 5, "Y": 7, "Z": 35},
    ]
    assert infer_for_bindings(blist) == ["X*Y=Z"]


def test_shapes_infer_for_bindings_splice_variables():
    blist: List[Bindings] = [{"X": 2, "YY": (3, 4)}, {"X": 5, "YY": (6, 4)}]
    assert infer_for_bindings(blist) == ["dims(YY)=2", "first(YY)=X+1", "last(YY)=4", "prod(YY)=4*X+4"]
    blist = [{"YY": (3, 4), "ZZ": (3, 4)}, {"YY": (6, 4), "ZZ": (6, 4)}]
    assert infer_for_bindings(blist) == [
        "ZZ=YY",
        "dims(YY)=2",
        "last(YY)=4",
    ]


def test_shape_inferrer_bindings():
    blist: List[Bindings] = [{"X": 2, "YY": (3, 4)}, {"X": 5, "YY": (6, 4)}]
    # Projections of the splice variable YY:
    assert ShapeInferrer(blist).bindings == [
        {"X": 2, "YY": (3, 4), "dims(YY)": 2, "first(YY)": 3, "last(YY)": 4, "prod(YY)": 12},
        {"X": 5, "YY": (6, 4), "dims(YY)": 2, "first(YY)": 6, "last(YY)": 4, "prod(YY)": 24},
    ]
    blist = [{"X": 2, "Y": 5}]
    # Product of the ordinary variables X and Y:
    assert ShapeInferrer(blist).bindings == [{"X": 2, "Y": 5, "X*Y": 10}]
    # Bindings with non-maximal chain length omitted, and chain length item discarded:
    blist = [{"X": 2, "Y": 5, CHAIN_LENGTH_KEY: 2}, {"X": 3, CHAIN_LENGTH_KEY: 1}]
    assert ShapeInferrer(blist).bindings == [{"X": 2, "Y": 5, "X*Y": 10}]


def alternator(arr):
    """
    Returns an object consisting of the even-numbered members of arr, whatever its type and dimensionality.
    Raises an exception if there is an odd number of members.
    """
    assert len(arr) % 2 == 0
    if isinstance(arr, list):
        return [arr[i] for i in range(0, len(arr), 2)]
    return arr[range(0, len(arr), 2)]


def test_shape_inference_process():
    arrays = [
        np.zeros((4)),
        [0] * 6,  # verify Shapes works with lists as well as numpy arrays
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),  # verify Shapes works with pandas DataFrames
        np.zeros((2, 5)),  # vary number of dimensions
        np.zeros((4, 3, 2)),
        np.zeros((3)),  # this one will fail on the assert in alternator, leaving an incomplete chain
    ]
    Shapes.clear()  # so we only get the bindings from the Shapes call below, not those from other tests
    for arr in arrays:
        try:
            Shapes(arr, "XX")(alternator(arr), "YY")
        except AssertionError:
            pass
    constraints = Shapes.infer()[1:]  # drop first (header) line
    endings = set(line.split(":")[-1].strip() for line in constraints)
    assert endings == {"dims(YY)=dims(XX)", "first(YY)=first(XX)/2", "prod(YY)=prod(XX)/2"}


def test_shapes_indexing():
    simple = [1, 2]
    s = Shapes(A2, "X")(simple, "Y")
    assert s[-1] == simple
    assert s.last() == simple


def test_evaluate_shape_expression():
    shapes = Shapes(A2, "X")(A24, "X,Y")
    assert shapes.evaluate_shape_expression(12) == 12
    assert shapes.evaluate_shape_expression(None) == "<<None>>"  # type: ignore
    assert shapes.evaluate_shape_expression("(X*Y)") == shapes.evaluate_shape_expression("X*Y")
    assert shapes.evaluate_shape_expression("Y-X") == 2
    assert shapes.evaluate_shape_expression("Y%X") == 0
    assert shapes.evaluate_shape_expression("Y|X") == "4<<|>>2"
