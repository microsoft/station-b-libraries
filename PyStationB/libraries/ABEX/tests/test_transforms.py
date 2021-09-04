# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
from abex.common.test_utils import assert_unordered_frame_equal
from abex.transforms import (
    CombinedTransform,
    InvertibleCombinedTransform,
    InvertibleTransform,
    LogTransform,
    MaxNormaliseTransform,
    RescaleTransform,
    ShiftTransform,
    Transform,
)

# TODO: test if name-clash raises backward


@pytest.fixture(autouse=True)
def dummy_df():
    data = {"a": [1, 0.2, 3, 48, 5], "b": [-3, -1, 0, 4.1, 4000], "c": [-100, 100, 0, 100, 5.0]}
    return pd.DataFrame(data)


class TestRescaleTransform:
    @pytest.fixture(autouse=True)
    def _dummy_input(self):
        self.input_df = pd.DataFrame({"a": [1, 2, -2], "b": [1, 2, -2], "c": [-1, 0, -1]}, dtype=np.float64)
        self.columns_to_rescale = ["a"]
        self.columns_to_rescale_with = ["b"]
        # Expected output given that rescaling happens with above columns
        self.expected_output_df = pd.DataFrame(
            {"a / b": [1, 1, 1], "b": [1, 2, -2], "c": [-1, 0, -1]}, dtype=np.float64
        )
        # Expected output if the 'drop rescaling' flag is set to True
        self.expected_output_drop_rescaling_df = pd.DataFrame({"a / b": [1, 1, 1], "c": [-1, 0, -1]}, dtype=np.float64)

    def test_forward__with_given_input__gives_expected(self):
        transform = RescaleTransform(
            columns=self.columns_to_rescale, rescaling_columns=self.columns_to_rescale_with, keep_rescaling_col=True
        )
        output_df = transform(self.input_df)
        assert_unordered_frame_equal(output_df, self.expected_output_df)

        # Test with the keep_rescaling_col flag set to False
        transform = RescaleTransform(
            columns=self.columns_to_rescale, rescaling_columns=self.columns_to_rescale_with, keep_rescaling_col=False
        )
        output_df = transform(self.input_df)
        assert_unordered_frame_equal(output_df, self.expected_output_drop_rescaling_df)


class TestShiftTransform:
    @pytest.fixture(autouse=True)
    def _dummy_input(self):
        self.input_df = pd.DataFrame({"a": [6, 20], "b": [1, 0]}, dtype=np.float64)
        self.columns_to_shift = ["a"]
        self.offsets = [-5]
        # Expected output given settings above
        self.expected_output_df = pd.DataFrame({"(a - 5)": [1, 15], "b": [1, 0]}, dtype=np.float64)
        # Expected output if scales inferred from df (using from_df constructor)
        self.expected_scales_from_df = [-6]
        self.expected_output_from_df_df = pd.DataFrame({"(a - 6)": [0, 14], "b": [1, 0]}, dtype=np.float64)

    def test_forward__with_given_input__gives_expected(self):
        transform = ShiftTransform(columns=self.columns_to_shift, offsets=self.offsets)
        output_df = transform(self.input_df)
        assert_unordered_frame_equal(output_df, self.expected_output_df)

    def test_forward__with_given_input_from_df__gives_expected(self):
        # Initialise the Normalise transform from a dataframe instead (infer the scales)
        transform = ShiftTransform.from_df(columns=self.columns_to_shift, df=self.input_df)
        # Assert the inferred scale is as expected
        assert self.expected_scales_from_df[0] == transform.offsets[0]
        # Assert output matches exptected
        output_df = transform(self.input_df)
        assert_unordered_frame_equal(output_df, self.expected_output_from_df_df)

    def test_prev_col_name(self):
        transform = ShiftTransform(columns=self.columns_to_shift, offsets=self.offsets)
        assert transform.prev_col_name("(a - 5)") == "a"
        # Gibberish should throw
        with pytest.raises(AssertionError):
            transform.prev_col_name("sdfghjk")
        # 'b' is an input column, but not one that's being shifted. Should throw
        with pytest.raises(AssertionError):
            transform.prev_col_name("(b - 5)")
        # 'a' is an input column that is to be shifted, but not an output column
        with pytest.raises(AssertionError):
            transform.prev_col_name("a")


class TestMaxNormaliseTransform:
    @pytest.fixture(autouse=True)
    def _dummy_input(self):
        self.input_df = pd.DataFrame({"a": [0, 20], "b": [1, 0]}, dtype=np.float64)
        self.columns_to_normalise = ["a"]
        self.scales = [5]
        # Expected output given settings above
        self.expected_output_df = pd.DataFrame({"a / 5": [0, 4], "b": [1, 0]}, dtype=np.float64)
        # Expected output if scales inferred from df (using from_df constructor)
        self.expected_scales_from_df = [20]
        self.expected_output_from_df_df = pd.DataFrame({"a / 20": [0.0, 1], "b": [1, 0]}, dtype=np.float64)

    def test_forward__with_given_input__gives_expected(self):
        transform = MaxNormaliseTransform(columns=self.columns_to_normalise, norm_scales=self.scales)
        output_df = transform(self.input_df)
        assert_unordered_frame_equal(output_df, self.expected_output_df)

    def test_forward__with_given_input_from_df__gives_expected(self):
        # Initialise the Normalise transform from a dataframe instead (infer the scales)
        transform = MaxNormaliseTransform.from_df(columns=self.columns_to_normalise, df=self.input_df)
        # Assert the inferred scale is as expected
        assert self.expected_scales_from_df[0] == transform.norm_scales[0]
        # Assert output matches exptected
        output_df = transform(self.input_df)
        assert_unordered_frame_equal(output_df, self.expected_output_from_df_df)

    def test_prev_col_name(self):
        transform = MaxNormaliseTransform(columns=self.columns_to_normalise, norm_scales=self.scales)
        assert transform.prev_col_name("a / 5") == "a"
        # Gibberish should throw
        with pytest.raises(AssertionError):
            transform.prev_col_name("sdfghjk")
        # 'b' is an input column, but not one that's being normalised. Should throw
        with pytest.raises(AssertionError):
            transform.prev_col_name("b / 5")
        # 'a' is an input column that is to be normalised, but not an output column
        with pytest.raises(AssertionError):
            transform.prev_col_name("a")


class TestLogTransform:
    @pytest.fixture(autouse=True)
    def _dummy_input(self):
        self.input_df = pd.DataFrame({"a": [1, 10, 100], "b": [1, 0, -5]}, dtype=np.float64)
        self.columns_to_log = ["a"]
        self.offsets = [0]
        # Expected output given settings above
        self.expected_output_df = pd.DataFrame({"log(a)": [0, 1, 2], "b": [1, 0, -5]}, dtype=np.float64)

        # This time with an offset
        self.input_df2 = pd.DataFrame({"a": [2, 11, 101], "b": [1, 0, -5]}, dtype=np.float64)
        self.offsets2 = [-1]
        # Exptected output the same
        self.expected_output_df2 = self.expected_output_df

    def test_forward__with_given_input__gives_expected(self):
        transform = LogTransform(columns=self.columns_to_log, offsets=self.offsets)
        output_df = transform(self.input_df)
        assert_unordered_frame_equal(output_df, self.expected_output_df)

    def test_forward__with_given_input_with_offset__gives_expected(self):
        transform = LogTransform(columns=self.columns_to_log, offsets=self.offsets2)
        assert transform.offsets[0] == self.offsets2[0]
        output_df = transform(self.input_df2)
        assert_unordered_frame_equal(output_df, self.expected_output_df2)

    def test_drop_nonpositive__drops_expected(self):
        pass

    def test_prev_col_name(self):
        transform = LogTransform(columns=self.columns_to_log, offsets=self.offsets)
        assert transform.prev_col_name("log(a)") == "a"
        # Gibberish should throw
        with pytest.raises(AssertionError):
            transform.prev_col_name("sdfghjk")
        # 'b' is an input column, but not one that's logged. Should throw
        with pytest.raises(AssertionError):
            transform.prev_col_name("log(b)")
        # 'a' is an input column that is to be logged, but not an output column
        with pytest.raises(AssertionError):
            transform.prev_col_name("a")


class TestCombinedTransform:
    @pytest.mark.parametrize(
        "transform_list",
        [
            [
                MaxNormaliseTransform(columns=["a", "b"], norm_scales=[3, 5000]),
                MaxNormaliseTransform(columns=["a / 3", "b / 5e+03"], norm_scales=[3, 4]),
            ],
            [
                MaxNormaliseTransform(columns=["a", "b", "c"], norm_scales=[3, 5000, 1]),
                LogTransform(columns=["a / 3"], offsets=[0]),
            ],
            [
                ShiftTransform(columns=["a"], offsets=[3]),
                MaxNormaliseTransform(columns=["(a + 3)", "b", "c"], norm_scales=[3, 5000, 1]),
                LogTransform(columns=["(a + 3) / 3"], offsets=[0]),
            ],
            [
                LogTransform(columns=["a", "b"], offsets=[0, 30]),
                RescaleTransform(columns=["log(a)"], rescaling_columns=["log(b)"]),
            ],
        ],
    )
    def test_forward__same_as_applying_individually(self, transform_list, dummy_df):
        combined_transform = CombinedTransform(transform_list)
        combined_transformed_df = combined_transform(dummy_df)
        # Apply each transform individually
        individually_applied_df = dummy_df
        for trans in transform_list:
            individually_applied_df = trans(individually_applied_df)
        assert_unordered_frame_equal(individually_applied_df, combined_transformed_df)


@pytest.mark.parametrize(
    "invertible_transform",
    [
        MaxNormaliseTransform(columns=["a", "b"], norm_scales=[3, 5000]),
        ShiftTransform(columns=["a", "b", "c"], offsets=[-3, 5000, -1.324343]),
        LogTransform(columns=["a"], offsets=[0]),
        LogTransform(columns=["a", "b"], offsets=[0, 3.11]),
        InvertibleCombinedTransform(
            transforms=[
                MaxNormaliseTransform(columns=["a", "b"], norm_scales=[3, 3]),
                LogTransform(columns=["b / 3"], offsets=[1.0001]),
            ]
        ),
    ],
)
def test_forward_backward_same(invertible_transform: InvertibleTransform, dummy_df):
    transformed_df = invertible_transform(dummy_df)
    recovered_df = invertible_transform.backward(transformed_df)
    print(recovered_df)
    assert_unordered_frame_equal(dummy_df, recovered_df)


@pytest.mark.parametrize(
    "transform, nameclash_df",
    [
        (MaxNormaliseTransform(columns=["a"], norm_scales=[3]), pd.DataFrame({"a": [1], "a / 3": [5]})),
        (LogTransform(columns=["a"], offsets=[0]), pd.DataFrame({"a": [5, 10], "b": [4, 1], "log(a)": [0, 0]})),
        (
            RescaleTransform(columns=["a"], rescaling_columns=["b"]),
            pd.DataFrame({"a": [5, 10], "b": [4, 1], "a / b": [0, 0]}),
        ),
        (
            InvertibleCombinedTransform(
                transforms=[
                    MaxNormaliseTransform(columns=["a"], norm_scales=[3]),
                    LogTransform(columns=["a / 3"], offsets=[-100]),
                ]
            ),
            pd.DataFrame({"a": [1], "log(a / 3)": [5]}),
        ),
    ],
)
def test_name_clash_raises(transform: Transform, nameclash_df: pd.DataFrame):
    with pytest.raises(ValueError):
        transform(nameclash_df)


@pytest.mark.parametrize(
    "transform, nameclash_df",
    [
        (MaxNormaliseTransform(columns=["a"], norm_scales=[3]), pd.DataFrame({"a": [1], "a / 3": [5]})),
        (LogTransform(columns=["a"], offsets=[0]), pd.DataFrame({"a": [5, 10], "b": [4, 1], "log(a)": [0, 0]})),
        (
            InvertibleCombinedTransform(
                transforms=[
                    ShiftTransform(columns=["a"], offsets=[3]),
                    LogTransform(columns=["a + 3"], offsets=[-100]),
                ]
            ),
            pd.DataFrame({"a": [1], "log(a + 3)": [5]}),
        ),
    ],
)
def test_name_clash_raises_backward(transform: InvertibleTransform, nameclash_df: pd.DataFrame):
    with pytest.raises(ValueError):
        transform.backward(nameclash_df)


def test_empty_compose():
    pass
