# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Data loading and processing functionality.

The general aim in this module is to do all data processing with
pandas and numpy, and NOT converting to tensors. As such, there
should be no reference to PyTorch, TensorFlow, etc. This will enable
switching frameworks later on, if necessary.

Exports:
    Dataset: the class that stores the data
"""
import functools
import itertools
import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from abex.constants import FILE
from abex.data_settings import DataSettings, NormalisationType
from abex.transforms import (
    CombinedTransform,
    InvertibleCombinedTransform,
    LogTransform,
    MaxNormaliseTransform,
    OneHotTransform,
    RescaleTransform,
    ShiftTransform,
)

# noinspection PyPep8Naming
from psbutils.arrayshapes import Shapes

# noinspection PyTypeHints
pd.options.mode.chained_assignment = None  # type: ignore


class Dataset:
    """A dataset class to store and preprocess data for Bayesian Optimization.

    It takes an unprocessed DataFrame with continuous inputs, (possible) categorical inputs and output(s),
    creates preprocessing transforms as specified in config (log, normalise e.t.c.), applies the preprocessing
    transforms to the dataframe, and saves the processed data as numpy arrays to which a model can be fit.

    The categorical inputs (if any given) are converted to the real domain with a one-hot encoding.

    The transforms are stored in self.preprocessing_transforms and self.onehot_transform, allowing for recovering
    the values in the original (pretransformed) space from those in model space (e.g. when generating a batch).

    The input DataFrame must include all columns specified by data_settings.inputs, data_settings.categorical_inputs,
    data_settings.output_column and in addition must have a column named "File"
    """

    def __init__(self, df: pd.DataFrame, data_settings: DataSettings) -> None:
        """Convert Pandas DataFrame into a local Dataset.

        Args:
            df: pandas data frame to be stored
            data_settings: object specifying e.g. what columns should be used and how they should be transformed
        """
        orig_len = len(df)
        df_pruned = df.dropna()
        new_len = 0 if df_pruned is None else len(df_pruned)
        if new_len < orig_len:
            logging.warning(f"Dropped {orig_len - new_len} of {orig_len} rows with N/A values")  # pragma: no cover

        if not new_len:  # pragma: no cover
            raise ValueError("The processed DataFrame is empty. Non-empty data is required to construct a Dataset.")
        assert df_pruned is not None

        # Store input/output names from data config
        self.pretransform_input_names: List[str] = data_settings.input_names  # Continuous inputs
        self.pretransform_categ_input_names: List[str] = data_settings.categorical_inputs  # Categorical inputs
        self.pretransform_output_name: str = data_settings.output_column

        # Store the dataframe
        self.pretransform_df = df_pruned.copy()
        # Get the combined preprocessing data transform
        self.preprocessing_transform: CombinedTransform = self.make_preprocessing_transforms(
            df=df_pruned, data_settings=data_settings
        )
        # Transform the data
        self.transformed_df: pd.DataFrame = self.preprocessing_transform(df_pruned)

        # If any categorical inputs given, make onehot_transform
        self.onehot_transform: Optional[OneHotTransform] = None
        self.onehot_inputs_array: Optional[np.ndarray] = None

        if self.transformed_categ_input_names:
            # Make the onehot transform
            # TODO: Allow to specify categorical space in config
            self.onehot_transform = OneHotTransform.from_df(data_settings.categorical_inputs, df=self.transformed_df)
            self.onehot_inputs_array = self.onehot_transform(self.transformed_df)[  # type: ignore # auto
                self.onehot_column_names
            ].values  # type: ignore # auto

        # Save the categorical inputs dataframe (useful for conditioning on a particular categorical value)
        self.categorical_inputs_df: pd.DataFrame = self.transformed_df[  # type: ignore # auto
            self.transformed_categ_input_names
        ]  # type: ignore # auto
        # Generate (transformed) data arrays
        self.continuous_inputs_array: np.ndarray = self.transformed_df[  # type: ignore # auto
            self.transformed_cont_input_names
        ].values  # type: ignore # auto
        self.inputs_array: np.ndarray = (  # type: ignore # auto
            np.concatenate([self.continuous_inputs_array, self.onehot_inputs_array], axis=1)
            if self.onehot_inputs_array is not None
            else self.continuous_inputs_array
        )
        self.output_array: np.ndarray = self.transformed_df[  # type: ignore # auto
            self.transformed_output_name
        ].values  # type: ignore # auto
        Shapes(self.inputs_array, "X,Y")(self.output_array, "X")  # type: ignore # auto

        # Get (continuous) parameter bounds in the original and transformed spaces
        self.pretransform_cont_param_bounds = data_settings.get_bounds_from_config_and_data(df)
        self.continuous_param_bounds = self.get_transformed_bounds(self.pretransform_cont_param_bounds)

        self.file: Optional[np.ndarray] = None
        if FILE in df_pruned.columns:
            self.file = df[FILE].values  # type: ignore # auto
        self.unique_files: List[str] = list(data_settings.files.keys())
        if len(self) == 0:  # pragma: no cover
            raise ValueError("No data-points returned after pre-processing. Exiting...")
        logging.info(f"- Created dataset with {len(self)} data points")

    def __len__(self) -> int:
        """Number of examples in the post-processed arrays (and dataframe)."""
        return self.continuous_inputs_array.shape[0]

    @classmethod
    def from_data_settings(cls, data_settings: DataSettings):
        """
        :param data_settings: a DataSettings object
        :return: the Dataset created by loading a dataframe from it
        """

        df = data_settings.load_dataframe()
        return Dataset(df, data_settings)

    @property
    def transformed_cont_input_names(self) -> List[str]:
        """Post-processed column names of the continuous inputs in the dataset."""
        return self.preprocessing_transform.transformed_col_names(self.pretransform_input_names)

    @property
    def transformed_categ_input_names(self) -> List[str]:
        """Post-processed column names of the categorical inputs in the dataset. (before one-hot)"""
        # Assumes there are no transforms for categorical inputs (yet)
        return self.pretransform_categ_input_names

    @property
    def onehot_column_names(self) -> List[str]:
        """Returns a list of column names for the one-hot encoded categorical variables. I.e. if the category columns
        have names ['Col1', 'Col2'...] and each of the inputs in each columns takes on values in some categorical space,
        then the resulting columns names will be: ['Col1:label11', 'Col1:label12', ..., 'Col2:label21', ...]
        where 'labelXX' are the values that inputs in that column can take.
        """
        if self.onehot_transform is None:
            return []
        return self.onehot_transform.onehot_column_names

    @property
    def transformed_input_names(self) -> List[str]:
        """Return the column names for the entire post-processed dataset including onehot inputs
        (each value of the onehot representation will be given its own column name).
        """
        return self.transformed_cont_input_names + self.onehot_column_names

    @property
    def transformed_output_name(self) -> str:
        """The name of the output signal after a transform has been applied"""
        return self.preprocessing_transform.transformed_col_names([self.pretransform_output_name])[0]

    @property
    def n_continuous_inputs(self) -> int:  # pragma: no cover
        """The number of continuous-valued input variables"""
        return len(self.transformed_cont_input_names)

    @property
    def n_categorical_inputs(self) -> int:  # pragma: no cover
        """The number of categorical-valued input variables"""
        return len(self.transformed_categ_input_names)

    @property
    def categorical_param_space(self) -> "OrderedDict[str, List]":
        # TODO: 'OrderedDict' above is a work-around. This will work with mypy in Python 3.8
        """Produce a dictionary containing the values associated with each categorical variable"""
        return self.onehot_transform.categorical_space if self.onehot_transform else OrderedDict()

    @property
    def ndims(self) -> int:
        """Return the number of input dimensions in the final post-processed dataset (including one-hot encodings of
        categorical variables).

        Ex. if the postprocessed dataset has M continuous inputs and N categorical inputs with C categories each,
        method would return M + N * C
        """
        return self.inputs_array.shape[1]

    @staticmethod
    def make_preprocessing_transforms(df: pd.DataFrame, data_settings: DataSettings) -> CombinedTransform:
        """Create preprocessing transforms for the inputs and the output as dictated by the config, and return
        a (possibly invertible) CombinedTransform that applies each of the preprocessing transform sequentially.

        If rescaling by a column is specified as a preprocessing step for any of the variables, the transform will not
        be invertible, meaning a Bayes.Opt. batch cannot be generated in the original space.
        """
        combined_transform: CombinedTransform = InvertibleCombinedTransform(transforms=[])

        # -- Input transforms
        # - Rescale inputs
        # Get the inputs which are to be rescaled
        inputs_to_rescale = [
            input_col
            for input_col in data_settings.inputs.keys()
            if data_settings.inputs[input_col].rescale_column is not None
        ]
        if inputs_to_rescale:  # pragma: no cover
            # Get the corresponding columns to rescale by
            rescale_columns = list(
                map(lambda input_name: data_settings.inputs[input_name].rescale_column, inputs_to_rescale)
            )
            combined_transform = combined_transform.extend(
                RescaleTransform(
                    columns=combined_transform.transformed_col_names(inputs_to_rescale),
                    rescaling_columns=rescale_columns,  # type: ignore
                )
            )

        # - Log-transform inputs:
        # Get the inputs to log
        inputs_to_log = [
            input_col for input_col in data_settings.inputs.keys() if data_settings.inputs[input_col].log_transform
        ]
        if inputs_to_log:
            # Get the corresponding offsets
            offsets = list(map(lambda input_name: data_settings.inputs[input_name].offset, inputs_to_log))
            combined_transform = combined_transform.extend(
                LogTransform(columns=combined_transform.transformed_col_names(inputs_to_log), offsets=offsets)
            )
        # - "final" (post-log) shift-transform on inputs
        inputs_to_offset = [
            input_col
            for input_col in data_settings.inputs.keys()
            if data_settings.inputs[input_col].final_offset != 0.0
        ]
        if inputs_to_offset:  # pragma: no cover
            # Get the corresponding offsets
            offsets = list(map(lambda input_name: data_settings.inputs[input_name].final_offset, inputs_to_offset))
            combined_transform = combined_transform.extend(
                ShiftTransform(columns=combined_transform.transformed_col_names(inputs_to_offset), offsets=offsets)
            )

        # - Normalise inputs
        # - If doing 'full-normalisation' onto [0, 1] range, first shift inputs by subtracting the minimum
        inputs_to_shift = [
            input_col
            for input_col in data_settings.inputs.keys()
            if data_settings.inputs[input_col].normalise == NormalisationType.FULL
        ]
        if inputs_to_shift:
            # Get the lower bounds of the input optimization space for the inputs to normalise (if those are given)
            bounds_minima = [
                data_settings.inputs[input_name].lower_bound or df[input_name].min() for input_name in inputs_to_shift
            ]
            transformed_bounds_minima = combined_transform(pd.DataFrame([bounds_minima], columns=inputs_to_shift))
            combined_transform = combined_transform.extend(
                ShiftTransform(
                    columns=combined_transform.transformed_col_names(inputs_to_shift),
                    offsets=-transformed_bounds_minima.to_numpy().ravel(),  # type: ignore
                )
            )
        # If doing either 'full' normalisation onto [0, 1] range, or just 'max' normalisation, divide by the maximum
        inputs_to_max_normalise = [
            input_col
            for input_col in data_settings.inputs.keys()
            if data_settings.inputs[input_col].normalise in [NormalisationType.FULL, NormalisationType.MAX_ONLY]
        ]
        if inputs_to_max_normalise:
            # Get the upper bounds of the input optimization space for the inputs to normalise (if given)
            bounds_maxima = [
                data_settings.inputs[input_name].upper_bound or df[input_name].max()
                for input_name in inputs_to_max_normalise
            ]
            transformed_bounds_maxima = combined_transform(
                pd.DataFrame([bounds_maxima], columns=inputs_to_max_normalise)
            )
            combined_transform = combined_transform.extend(
                MaxNormaliseTransform(
                    columns=combined_transform.transformed_col_names(inputs_to_max_normalise),
                    norm_scales=transformed_bounds_maxima.to_numpy().ravel(),  # type: ignore
                )
            )

        # -- Output transforms (note they will look slightly different if a multi-output option is added)
        # - Rescale output
        if data_settings.output_settings.rescale_column:
            combined_transform = combined_transform.extend(  # pragma: no cover
                RescaleTransform(
                    columns=[data_settings.output_column],
                    rescaling_columns=[data_settings.output_settings.rescale_column],
                )
            )

        # - Log-transform output:
        if data_settings.output_settings.log_transform:
            combined_transform = combined_transform.extend(
                LogTransform(
                    columns=combined_transform.transformed_col_names([data_settings.output_column]),
                    offsets=[data_settings.output_settings.offset],
                )
            )
        # - "final" (post-log) shift-transform on output
        if data_settings.output_settings.final_offset:
            # Get the corresponding offsets
            combined_transform = combined_transform.extend(  # pragma: no cover
                ShiftTransform(
                    columns=combined_transform.transformed_col_names([data_settings.output_column]),
                    offsets=[data_settings.output_settings.final_offset],
                )
            )

        # - Normalise output:
        # - If doing 'full-normalisation' onto [0, 1] range, first shift inputs by subtracting the minimum
        if data_settings.output_settings.normalise == NormalisationType.FULL:
            combined_transform = combined_transform.extend(  # pragma: no cover
                ShiftTransform.from_df(
                    columns=combined_transform.transformed_col_names([data_settings.output_column]),
                    df=combined_transform(df),
                )
            )
        # If doing either 'full' normalisation onto [0, 1] range, or just 'max' normalisation, divide by the maximum
        if data_settings.output_settings.normalise in [NormalisationType.FULL, NormalisationType.MAX_ONLY]:
            combined_transform = combined_transform.extend(
                MaxNormaliseTransform.from_df(
                    columns=combined_transform.transformed_col_names([data_settings.output_column]),
                    df=combined_transform(df),
                )
            )

        # TODO: By this point, combined_transform(df) has been called twice. Return it for comp. efficiency?
        return combined_transform

    def get_transformed_bounds(
        self, pretransform_bounds: "OrderedDict[str, Tuple[float, float]]"
    ) -> "OrderedDict[str, Tuple[float, float]]":
        """Return a dictionary containing the bounds of each input variable in their original units"""
        # Transform into an ordered dictionary mapping transformed_column_name to a tuple of [lower_bound, upper_bound]
        transformed_bounds = self.preprocessing_transform(pd.DataFrame(pretransform_bounds))
        transformed_bounds_ordered = OrderedDict(
            (col, transformed_bounds[col]) for col in self.transformed_cont_input_names
        )
        return transformed_bounds_ordered  # type: ignore

    @property
    def parameter_space(self) -> "Tuple[OrderedDict[str, Tuple[float, float]], OrderedDict[str, List]]":
        """Return a tuple of OrderedDicts representing the parameter-space for the continuous and categorical variables
        respectively.

        This is a wrapper around self.continuous_param_bounds and self.categorical_param_space.
        """
        return self.continuous_param_bounds, self.categorical_param_space

    @staticmethod
    def merge(datasets: List["Dataset"]) -> "Dataset":  # pragma: no cover
        """Merges a few data sets together.

        Args:
            datasets: list of data sets to be merged

        Returns:
            a merged data set

        Todo:
            TODO: This method is not implemented -- if there are more than one data set, a NotImplementedError
             is raised
            TODO: The API of this method may change -- different data sets may have different columns or
             different transforms to be used. We should think what is the expected behaviour.
        """
        if len(datasets) == 1:
            return datasets[0]
        else:
            raise NotImplementedError("Not implemented merging of multiple datasets yet.")

    @staticmethod
    def _decide_indices(size, randomize: bool) -> np.ndarray:  # pragma: no cover
        """Get an array of indices for all examples in the dataset. If randomize=True, the order will be randomized,
        otherwise the indices will be arranged sequentially.
        """
        # TODO: this method isn't used anywhere. It could be used below in split_folds?
        if randomize:
            return np.random.permutation(size)
        return np.arange(size, dtype=int)

    def split_folds(self, num_folds: int, seed: Optional[int] = None) -> List[np.ndarray]:  # pragma: no cover
        """Generate a list of length num_folds with arrays of indices pointing to the examples to allocate to each
        fold."""
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.permutation(len(self))
        return np.array_split(indices, num_folds)

    def generate_subset(self, ids) -> "Dataset":
        """Generates a new data set representing a subset of a previously instantiated Dataset. The bounds
        and transforms are retained from the original dataset, which makes any models defined on the original instance
        and its subset-based derivatives be compatible.

        Args:
            ids: list or numpy array of integers which can be used to retrieve samples from pandas data frame

        Returns:
            A new Dataset instance that represents a subset of a previously instantiated Dataset.

        Todo:
            This method uses `deepcopy` and should be refactored. This however involves non-trivial changes, as a new
            constructor needs to be developed.
        """
        d = deepcopy(self)  # TODO: using deepcopy strongly discouraged by Guido van Rossum
        d.pretransform_df = self.pretransform_df.iloc[ids, :]
        d.transformed_df = self.transformed_df.iloc[ids, :]  # type: ignore
        d.categorical_inputs_df = self.categorical_inputs_df.iloc[ids, :]  # type: ignore
        d.continuous_inputs_array = self.continuous_inputs_array[ids]
        d.onehot_inputs_array = self.onehot_inputs_array[ids] if self.onehot_inputs_array is not None else None
        d.inputs_array = self.inputs_array[ids]
        d.output_array = self.output_array[ids]
        if d.file is not None:
            d.file = d.file[ids]
        return d

    def train_test_split(self, test_ids) -> Tuple["Dataset", "Dataset"]:
        """Returns a training and a test data set.

        Args:
            test_ids: list or numpy array of integers, which can be used to retrieve samples from a pandas data frame

        Returns:
            training data set, corresponding to all samples *other* than indexed by `test_ids`
            test data set, corresponding to samples indexed by `test_ids`

        Related:
            generate_subset
        """
        # create list of indices for train and test data sets
        all_ids = np.arange(len(self), dtype=int)
        train_ids = np.setdiff1d(all_ids, test_ids)
        train = self.generate_subset(train_ids)
        test = self.generate_subset(test_ids)
        return train, test

    def train_test_split_per_file(
        self, test_file_ids: Iterable[Union[str, int]]
    ) -> Tuple["Dataset", Optional["Dataset"]]:
        """Returns a training and a test data set. Instead of heldout by indexes, heldout by file ids

        Args:
            test_file_ids: list or numpy array of strings/integers corresponding to file ids

        Returns:
            training dataset, corresponding to all samples whose corresponding file id is *other*
            than those in `test_file_ids`
            test data set (or None if self.file is None), corresponding to samples whose file ids belong
            to `test_file_ids`

        Related:
            train_test_split, generate_subset
        """
        if self.file is None:
            return self, None  # pragma: no cover
        else:
            test_ids = [i for i, fid in enumerate(self.file) if fid in test_file_ids]
            return self.train_test_split(test_ids)

    @property
    def n_unique_contexts(self) -> int:  # pragma: no cover
        """Return the number of possible combinations of categorical variables. If no categorical variables in dataset,
        return 0

        Example:
            If categorical_param_space is:
                {'colour': [red, blue],
                 'number': [1, 2, 3]}
            The number of possible combinations of 'colour' and 'number' is 2*3 = 6
        """
        if self.n_categorical_inputs == 0:
            return 0  # pragma: no cover
        # Calculate the size of the categorical space for each categorical variable
        n_categories_for_input = [len(possible_categs) for possible_categs in self.categorical_param_space.values()]
        # Return the size of the Cartesian product of these spaces
        return functools.reduce(lambda a, b: a * b, n_categories_for_input)

    def unique_context_generator(self) -> Generator:
        """Returns a generator of all possible unique categorical contexts - i.e. unique combinations of categorical
        variables. This is useful when creating for-loops over all possible combinations of such variables.

        Example:
            If categorical_param_space is:
                {'colour': [red, blue],
                 'number': [1, 2, 3]}
            this generator would return:
                [red, 1]
                [red, 2]
                ...
                [blue, 2]
                [blue, 3]

        Yields:
            List: A list of length self.n_categorical_inputs with a particular combination of categories
        """
        if self.n_categorical_inputs == 0:  # pragma: no cover
            raise Exception("There are no categorical variables in dataset, hence no contexts to iterate over.")
        for context in itertools.product(*self.categorical_param_space.values()):  # pragma: no cover
            yield context

    def unique_context_generator_or_none(self) -> Generator:  # pragma: no cover
        """Wrapper around self.unique_context_generator to return Generator([None]) if no categorical inputs are
        present in the dataset.
        """
        if self.n_categorical_inputs == 0:
            yield None  # pragma: no cover
        else:
            yield from self.unique_context_generator()

    def categ_context_to_onehot(self, context: List) -> np.ndarray:  # pragma: no cover
        """Converts the categorical variable assignments in a 'context' (emukit) to a one-hot encoding. A context is used
        to fix variables and thus reduce the dimensionality in a decision space, during Bayesian optimization. N.B.
        all categorical variables must be assigned in the context to create a valid one-hot encoding.

        Args:
            context: A dictionary storing variable assignments

        Returns:
            numpy array containing one-hot encoding of the categorical variables
        """
        assert self.n_categorical_inputs > 0, (
            f"One-hot encoding only available if there are any categorical "
            f"inputs. In this dataset there are: {self.n_categorical_inputs}."
        )

        assert len(context) == self.n_categorical_inputs, (
            f"Mismatch between the length of the context and the "
            f"number of categorical inputs: {len(context)} != "
            f"{self.n_categorical_inputs}."
        )
        assert self.onehot_transform is not None, "One-hot transform needs to be set for categorical inputs."
        context_df = pd.DataFrame({self.transformed_categ_input_names[i]: [context[i]] for i in range(len(context))})
        onehot_enc = self.onehot_transform(context_df)
        return onehot_enc.values.ravel()

    def filtered_on_context(self, context_dict: Dict[str, Any]) -> "Dataset":  # pragma: no cover
        """Return a sub-dataset with only the examples where the values match the values given in context.

        Args:
            context_dict: A dictionary from 'column_names' -> 'context_values' to condition on. Only examples from this
                dataset for which values match the ones given in the context dictionary will be present in
                the returned dataset.

        Returns:
            A new (sub-)dataset with only the examples that match the values in context.
        """
        context_df = pd.DataFrame(context_dict, index=[0])
        transformed_context_series = self.preprocessing_transform(context_df).iloc[0]
        transformed_context_cols = transformed_context_series.index

        # Extract the idxs of locations where examples in self.transformed_df match the values of context
        locs = np.where((self.transformed_df[transformed_context_cols] == transformed_context_series).all(axis=1))[0]
        filtered_dataset = self.generate_subset(locs)
        return filtered_dataset

    def summarise_by_categories(self) -> pd.DataFrame:  # pragma: no cover
        """Produce a dataframe that counts how many data-points have a given combination of categorical values.

        For instance, in a DataFrame with two categories 'A' and 'B' (each with possible values in [1, 2, 3]), the
        output might look like:

        'A' | 'B' | 'Examples Count'
        '1'   '1'   $Num. examples with 'A'=='1' and 'B' == '1'
        '1'   '2'   $Num. examples with 'A'=='1' and 'B' == '2'
           ...
        '3'   '3'   $Num. examples with 'A'=='3' and 'B' == '3'

        TODO: Describe what is the format of the returned data frame. Is the index hierarchical or linear?
        """
        return (
            self.categorical_inputs_df.groupby(self.transformed_categ_input_names)
            .size()
            .reset_index()
            .rename(columns={0: "Example Count"})  # type: ignore # auto
        )

    def augment_single(self, locs, inpt=0.5):  # pragma: no cover
        """Add zero output values when essential inputs are zero.

        Todo:
            This method does not work at the current state and raises NotImplementedError when called.
        """
        # TODO: This method needs rewriting after the interface update
        raise NotImplementedError()
        n = len(locs)
        augment1_x = inpt * np.ones((n, self.n_inputs))
        for i, j in enumerate(locs):
            augment1_x[i, j] = 0.0
        for g, labels in zip(self.onehot_vectors, self.onehot_labels):
            axg = np.append(augment1_x, np.tile(g, (n, 1)), axis=1)
            self.x = np.append(self.x, axg, axis=0)
            self.y = np.append(self.y, self.zero * np.ones(n))
            self.n_data += n
            for key, cvalue in zip(self.c.keys(), labels):
                self.c[key] = np.append(self.c[key], np.tile(cvalue, n))
