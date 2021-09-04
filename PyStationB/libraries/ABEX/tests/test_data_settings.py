# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from abex.common.test_utils import assert_unordered_frame_equal
from abex.dataset import FILE, Dataset
from abex.data_settings import (
    DataSettings,
    ParameterConfig,
    InputParameterConfig,
    NormalisationType,
)
from abex.transforms import LogTransform, MaxNormaliseTransform, ShiftTransform
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from psbutils.psblogging import logging_to_stdout


@pytest.fixture
def example_data_config():
    config = DataSettings()
    config.inputs = {  # type: ignore
        "a": InputParameterConfig(unit="mg", normalise=NormalisationType.FULL),
        "b": InputParameterConfig(
            normalise=NormalisationType.NONE, log_transform=True, offset=4, lower_bound=-3, upper_bound=4
        ),
    }
    config.output_column = "Crosstalk Ratio"
    config.output_settings = ParameterConfig(unit="ml", normalise=NormalisationType.NONE)
    return config


class TestDataset:
    @pytest.fixture(autouse=True)
    def _dummy_inputs(self):
        # Define config for dummy input
        self.config1 = DataSettings()
        self.config1.inputs = {  # type: ignore
            "a": InputParameterConfig(unit="mg", normalise=NormalisationType.FULL, log_transform=False),
            "b": InputParameterConfig(
                normalise=NormalisationType.NONE, log_transform=True, offset=4, lower_bound=-3, upper_bound=4
            ),
        }
        self.config1.output_column = "y"
        self.config1.output_settings = ParameterConfig(
            unit="ml", normalise=NormalisationType.NONE, log_transform=True, offset=0
        )
        # Make dataframe for dummy input
        self.df1 = pd.DataFrame(
            {
                "a": [10.0, 6, 8],
                "b": [-3.0, 96, 996],
                "d": [0.1, 33, 2],
                "y": [100.0, 10, 1],
                FILE: ["somefile.csv"] * 3,
            }
        )
        self.df3 = pd.DataFrame(  # adding one dataframe with different file ids in it
            {
                "a": [10.0, 6, 8, 15],
                "b": [-3.0, 96, 996, 243],
                "d": [0.1, 33, 2, 12],
                "y": [100.0, 10, 1, 20.0],
                FILE: ["somefile.csv"] * 2 + ["file2.csv", "file3.csv"],
            }
        )

        # Expected post-transform df (excluding files and other columns)
        self.expected_transformed_df1 = pd.DataFrame(
            {"(a - 6) / 4": [1, 0.0, 0.5], "log(b)": [0, 2, 3], "log(y)": [2, 1, 0]}, dtype=np.float64
        )
        # Expected post-transform columns:
        self.expected_trans_cols1 = self.expected_transformed_df1.columns

    @pytest.fixture(autouse=True)
    def _dummy_inputs_with_categorical(self):
        # Define config for dummy input with categorical inputs
        self.config2 = DataSettings()
        self.config2.inputs = {  # type: ignore
            "a": InputParameterConfig(unit="mg", normalise=NormalisationType.NONE, log_transform=False, offset=0),
        }
        self.config2.output_column = "y"
        self.config2.output_settings = ParameterConfig(
            unit="ml", normalise=NormalisationType.NONE, log_transform=True, offset=0
        )
        self.config2.categorical_inputs = ["categ_col1", "categ_col2"]
        # Make dataframe for dummy input
        self.df2 = pd.DataFrame(
            {
                "a": [0.1, 1, 10],
                "categ_col1": ["a", "b", "a"],
                "categ_col2": [1, 2, 3],
                "d": [0.0, 33.0, 2],
                "y": [100.0, 10.0, 1],
                FILE: ["somefile.csv"] * 3,
            }
        )

    def test_init__instantiates_expected_transforms(self):
        dataset = Dataset(self.df1, self.config1)
        # Expected transforms are: Log on input, Shift and MaxNormalise on input, Log on output
        assert len(dataset.preprocessing_transform.transforms) == 4
        # Assert Log("b") transform is as expected from config
        assert isinstance(dataset.preprocessing_transform.transforms[0], LogTransform)
        assert ["b"] == dataset.preprocessing_transform.transforms[0].columns
        assert [self.config1.inputs["b"].offset] == dataset.preprocessing_transform.transforms[0].offsets
        # Assert Shift transform is as expected from config
        assert isinstance(dataset.preprocessing_transform.transforms[1], ShiftTransform)
        assert ["a"] == dataset.preprocessing_transform.transforms[1].columns
        # Assert MaxNormalise transform is as expected from config
        assert isinstance(dataset.preprocessing_transform.transforms[2], MaxNormaliseTransform)
        # Assert Log("y") (log output) transform is as expected from config
        assert isinstance(dataset.preprocessing_transform.transforms[3], LogTransform)
        assert ["y"] == dataset.preprocessing_transform.transforms[3].columns
        assert [self.config1.output_settings.offset] == dataset.preprocessing_transform.transforms[3].offsets

    def test_init__gives_expected_transformed_df(self):
        dataset = Dataset(self.df1, self.config1)
        for expected_col in self.expected_trans_cols1:  # type: ignore # auto
            assert expected_col in dataset.transformed_df.columns
        assert_frame_equal(dataset.transformed_df[self.expected_trans_cols1], self.expected_transformed_df1)

    def test_init__gives_expected_transformed_arrays(self):
        dataset = Dataset(self.df1, self.config1)
        # Checks on dimensions and other properties
        assert dataset.continuous_inputs_array.shape[1] == len(self.config1.input_names)
        assert dataset.output_array.ndim == 1
        assert dataset.output_array.shape[0] == dataset.continuous_inputs_array.shape[0]
        assert dataset.onehot_inputs_array is None
        # Numerically check values of output array
        assert_allclose(np.log10(self.df1["y"]), dataset.output_array.ravel())  # type: ignore # auto

    def test_init__with_categorical__gives_expected_transformed_arrays(self):
        dataset = Dataset(self.df2, self.config2)
        # Checks on dimensions and other properties
        assert dataset.continuous_inputs_array.shape[1] == len(self.config2.input_names)
        assert dataset.output_array.ndim == 1
        assert dataset.output_array.shape[0] == dataset.continuous_inputs_array.shape[0]
        assert dataset.onehot_inputs_array.shape[0] == dataset.continuous_inputs_array.shape[0]
        assert dataset.onehot_inputs_array.shape[1] == 5  # There are 2 + 3 unique categories in df

    def test_init__with_categorical__gives_expected_onehot_encoding(self):
        dataset = Dataset(self.df2, self.config2)
        for categ_col in self.config2.categorical_inputs:
            # Assert all the columns are present in parameter space:
            assert categ_col in dataset.categorical_param_space
            for value in self.df2[categ_col]:  # type: ignore # auto
                assert value in dataset.categorical_param_space[categ_col]
            assert len(dataset.categorical_param_space[categ_col]) == len(self.df2[categ_col].unique())
        # Assert onehot array matches expected shape:
        expected_onehot_dim = sum([len(self.df2[categ_col].unique()) for categ_col in self.config2.categorical_inputs])
        assert dataset.onehot_inputs_array.shape[1] == expected_onehot_dim
        # Assert all elements are 1 or 0
        assert np.all(np.isin(dataset.onehot_inputs_array, [0, 1]))
        # Assert the sum of each row is equal to the number of categorical columns
        assert np.all(dataset.onehot_inputs_array.sum(axis=1) == len(self.config2.categorical_inputs))

    def test_reverse_transform__gives_original_df(self):
        # Note: both dummy dataset are invertible
        # Without categorical inputs
        dataset = Dataset(self.df1, self.config1)
        assert hasattr(dataset.preprocessing_transform, "backward")
        reverse_pass_df = dataset.preprocessing_transform.backward(dataset.transformed_df)  # type: ignore
        assert_unordered_frame_equal(reverse_pass_df, self.df1)
        # Check with categorical inputs
        dataset = Dataset(self.df2, self.config2)
        reverse_pass_df = dataset.preprocessing_transform.backward(dataset.transformed_df)  # type: ignore
        assert_unordered_frame_equal(reverse_pass_df, self.df2)

    def test_generate_subset__subset_has_expected_fields(self):
        inds = [0, 1]
        dataset = Dataset(self.df1, self.config1)

        def check_subset_df_equals_dataset_chunk(subset, dataset, inds):
            """
            Check all fields are equal, except preprocessing_transform, continous_param_bounds,
            and categorical_inputs_df
            TODO: add check for 3 missing fields
            """
            assert subset.pretransform_df.equals(dataset.pretransform_df.iloc[inds, :])
            assert subset.transformed_df.equals(dataset.transformed_df.iloc[inds, :])
            assert np.array_equal(subset.continuous_inputs_array, dataset.continuous_inputs_array[inds, :])
            assert np.array_equal(subset.inputs_array, dataset.inputs_array[inds, :])
            assert np.array_equal(subset.output_array, dataset.output_array[inds])
            assert np.array_equal(subset.file, dataset.file[inds])
            for key in subset.__dict__.keys():
                print(key)
                if not (
                    key
                    in [
                        "pretransform_df",
                        "transformed_df",
                        "continuous_inputs_array",
                        "inputs_array",
                        "output_array",
                        "file",
                        "preprocessing_transform",
                        "categorical_inputs_df",
                        "continuous_param_bounds",
                    ]
                ):
                    assert subset.__dict__[key] == dataset.__dict__[key]

        # test with list as input
        subset_df = dataset.generate_subset(inds)
        check_subset_df_equals_dataset_chunk(subset_df, dataset, inds)
        # test with numpy as input
        subset_df2 = dataset.generate_subset(np.array(inds))
        check_subset_df_equals_dataset_chunk(subset_df2, dataset, inds)

    def test_standard_train_test_split(self):
        dataset = Dataset(self.df1, self.config1)
        _, dataset_test = dataset.train_test_split([0, 1])
        assert dataset_test.pretransform_df.equals(dataset.pretransform_df.iloc[[0, 1], :])

    def test_standard_train_test_split_per_file(self):
        dataset = Dataset(self.df3, self.config1)
        test_ids = ["somefile.csv"]
        dataset_train, dataset_test = dataset.train_test_split_per_file(test_ids)
        for x in list(np.unique(dataset_test.pretransform_df.File.values)):
            assert x in test_ids
        for x in list(np.unique(dataset_train.pretransform_df.File.values)):
            assert not (x in test_ids)

    def test_parameter_space__gives_expected(self):
        pass

    def test_emukit_parameter_space__gives_expected(self):
        pass


def test_drop_if_outside_bounds__drops_expected_rows(example_data_config: DataSettings):
    """Test that if drop outside bounds specified, the right examples are dropped"""
    param_config_input1 = InputParameterConfig(
        unit="mg",
        normalise=NormalisationType.FULL,
        lower_bounds=0,
        upper_bound=10.0,
        drop_if_outside_bounds=False,
    )
    param_config_input2 = InputParameterConfig(
        log_transform=True, offset=4, lower_bound=0, upper_bound=10.0, drop_if_outside_bounds=True
    )
    example_data_config.inputs = OrderedDict([("a", param_config_input1), ("b", param_config_input2)])
    # Test dataframe where:
    # 1st row: both 'a' and 'b' inside bounds (don't drop)
    # 2nd: only 'a' inside bounds - 'b' outside bounds (drop)
    # 3rd: only 'b' inside bounds - 'a' outside bounds (don't drop)
    # drop_if_outside_bounds should remove only the 2nd row.
    test_df = pd.DataFrame(
        {
            "a": [5.0, 1.0, 47.0],
            "b": [5.0, 46.0, 1.0],
            "y": [100.0, 100, 100],
            FILE: ["somefile.csv"] * 3,
        }
    )
    expected_output = test_df.iloc[[0, 2]].reset_index()  # type: ignore
    filtered_df = example_data_config.drop_if_outside_bounds(test_df)
    # Assert output is as expected
    assert_frame_equal(filtered_df, expected_output)
    assert len(filtered_df) == 2


def folders_from(location: Path, folder: Path):
    ds = DataSettings(config_file_location=location, folder=folder, simulation_folder=folder)
    return ds.locate_config_folders()


def test_locate_config_folders():
    logging_to_stdout()
    tmp_dir = TemporaryDirectory()
    tmp_dir_path = Path(tmp_dir.name)
    with (tmp_dir_path / "data.csv").open("w"):
        pass
    # Absolute folder path, config file location meaningless
    assert folders_from(Path("foo.yml"), tmp_dir_path) == [tmp_dir_path]
    # "folder" is relative to config file location
    tmp_cfg_path = tmp_dir_path / "foo.yml"
    tmp_folder_path = tmp_dir_path / "myfolder"
    tmp_folder_path.mkdir()
    with (tmp_folder_path / "data.csv").open("w"):
        pass
    assert folders_from(tmp_cfg_path, Path("myfolder")) == [tmp_folder_path]
    # "myfolder" is above config file location
    tmp_cfg_path = tmp_dir_path / "specs" / "foo.yml"
    assert folders_from(tmp_cfg_path, Path("myfolder")) == [tmp_folder_path]
    # "myfolder" is nowhere in relation to config file
    with pytest.raises(FileNotFoundError):
        folders_from(Path("foo.yml"), Path("myfolder"))
    tmp_dir.cleanup()
    with pytest.raises(FileNotFoundError):
        folders_from(tmp_cfg_path, Path("myfolder"))
