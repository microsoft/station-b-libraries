# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Module for defining transforms on data columns.

Exports:
    UnivariateFunctionTransform, a transform in which a single column at input is transformed into a single column at
        output, independently of the other columns to be transformed.
    CombinedTransform, combines multiple transforms into a single combined transform that applies them sequentially.
    InvertibleUnivariateFunctionTransform, an invertible UnivariateFunctionTransform
    InvertibleCombinedTransform, an investible CombinedTransform
    RescaleTransform, rescales a column by another specified column.
    LogTransform, takes the (base 10) logarithm of a column (with an optional additive pre-logged offset).
    ShiftTransform, shifts (adds a given offset to) all values in a column.
    MaxNormaliseTransform, divides all values in a column by a scale.
    OneHotTransform, converts a column containing categories to a onehot encoding.
"""
import abc
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class Transform(abc.ABC):
    """Abstract class defining the interface of a Transform class"""

    def __init__(self, columns: Sequence[str]) -> None:
        """
        Args:
            columns: columns in the DataFrame to be transformed by this transform
        """
        self._columns = list(columns)
        # Assert all columns unique
        assert len(self._columns) == len(set(self._columns)), f"Some columns in {self._columns} are not unique."

    @property
    def columns(self) -> Sequence[str]:
        """Access the columns"""
        return self._columns

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the transform to the (relevant columns) of the DataFrame. Wrapper around _forward(df) that validates
        the inputs before the transform is applied

        Args:
            df: DataFrame to apply transform to

        Returns:
            Transformed DataFrame
        """
        self._validate_df_for_forward(df)
        return self._forward(df)

    def _validate_df_for_forward(self, df: pd.DataFrame) -> None:
        """Validate that the transform can be applied to DataFrame given. Checks if there are any column name clashes
        if the transform were to be aplied.
        """
        pass_through_cols = set(df.columns) - set(self.columns)
        clashing_cols = set(self.transformed_col_names(self.columns)).intersection(pass_through_cols)
        if len(clashing_cols) != 0:
            raise ValueError(
                f"There is a name-clash between the input columns and transformed columns:\n{clashing_cols}"
            )

    @abc.abstractmethod
    def _forward(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        """Implements transforming the DataFrame"""
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.forward(df)

    @abc.abstractmethod
    def transformed_col_names(self, columns: Sequence[str]) -> List[str]:  # pragma: no cover
        """Get the names of the columns after the transform is be applied to a DataFrame with columns
        given in columns.
        """
        pass


class UnivariateFunctionTransform(Transform, metaclass=abc.ABCMeta):
    """Abstract class for a transform in which a single column at input is transformed into a single column at output,
    independently of the other columns to be transformed.
    """

    @abc.abstractmethod
    def new_col_name(self, column: str) -> str:  # pragma: no cover
        """Generate a new column name that incorporates details of the transformation."""
        pass

    def transformed_col_name(self, column: str) -> str:
        """Return the transformed column name."""
        return self.new_col_name(column) if column in self.columns else column

    def transformed_col_names(self, columns: Sequence[str]) -> List[str]:
        """Return the multiple transformed column names."""
        return [self.transformed_col_name(col) for col in columns]


class CombinedTransform(Transform):
    """Combines multiple transforms into a single combined transform that applies them sequentially."""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        """
        Args:
            transforms, a sequence of Transforms to combine
        """
        self.transforms = transforms

    @property
    def columns(self) -> List[str]:
        """Get the input column names of the columns that will be passed through this combined transform. Any of these
        columns will at some point be modified by at least one transform in self.transforms.

        Returns:
            A list of column names that will get transformed when passed through this combined transform.
        """
        input_columns = set()
        output_cols_so_far: Sequence[str] = list()
        for transform in self.transforms:
            # Get the new inputs columns that are not simply the output of one of the previous transforms
            new_input_cols = set(transform.columns) - set(output_cols_so_far)
            # Update input columns with those
            input_columns |= new_input_cols
            # Update the list of the transformed column names so-far
            output_cols_so_far = transform.transformed_col_names(list(output_cols_so_far) + list(new_input_cols))
        return list(input_columns)

    def extend(self, other: Transform) -> "CombinedTransform":  # pragma: no cover
        """Add another transform to the list of transforms applied in sequence by this combined transform. The new
        transform will be applied last in the sequence. Returns a new CombinedTransform object.
        """
        if isinstance(other, CombinedTransform):
            return CombinedTransform(list(self.transforms) + list(other.transforms))
        elif isinstance(other, Transform):
            return CombinedTransform(list(self.transforms) + [other])
        else:
            raise TypeError(
                f"Can't extend a combined transform with an object of type {type(other)}. Must be a Transform"
            )

    def _forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run df through each transform in turn and return the result.
        """
        for transform in self.transforms:
            df = transform.forward(df)
        return df

    def transformed_col_names(self, columns: Sequence[str]) -> List[str]:
        """
        Returns the transformed column names after passing through each of the transforms sequentially
        """
        transformed_cols = list(columns)
        for transform in self.transforms:
            transformed_cols = transform.transformed_col_names(transformed_cols)
        return transformed_cols


class InvertibleTransform(Transform, metaclass=abc.ABCMeta):
    """An abstract class defining the interface of a transform which can be inverted. For an invertible transform,
    the examples in the original space can be recovered by calling transform.backward().
    """

    def backward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse-transform the DataFrame."""
        self._validate_df_for_backward(df)
        return self._backward(df)

    def _validate_df_for_backward(self, df: pd.DataFrame) -> None:
        """
        Check that a DataFrame can be safely inverted by this transform.
        """
        pass_through_cols = set(df.columns) - set(self.transformed_col_names(self.columns))
        clashing_cols = set(self.columns).intersection(pass_through_cols)
        if len(clashing_cols) != 0:
            raise ValueError(
                f"There is a name-clash between input columns and inverse-transformed columns:\n{clashing_cols}"
            )

    @abc.abstractmethod
    def _backward(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        """Apply InvertibleTransform in reverse to supplied DataFrame"""
        pass


class InvertibleUnivariateFunctionTransform(InvertibleTransform, UnivariateFunctionTransform, metaclass=abc.ABCMeta):
    """An invertible UnivariateFunctionTransform. See parent classes for details."""

    @abc.abstractmethod
    def prev_col_name(self, column: str) -> str:  # pragma: no cover
        """Return the original (pre-transformed) column name"""
        pass


class InvertibleCombinedTransform(CombinedTransform, InvertibleTransform):
    """An investible CombinedTransform. See parent classes for details."""

    def __init__(self, transforms: List[InvertibleTransform]) -> None:
        """
        Args:
            transforms, a sequence of InvertibleTransforms to combine
        """
        self.transforms: List[InvertibleTransform] = transforms

    def extend(self, other: Transform) -> "CombinedTransform":
        """Add another transform to the list of transforms applied in sequence by this combined transform. The new
        transform will be applied last in the sequence.

        Returns a new InvertibleCombinedTransform object if the new transform is invertible, or a CombinedTransform
        object if the new transform is not invertible.
        """
        if isinstance(other, InvertibleCombinedTransform):
            return InvertibleCombinedTransform(self.transforms + other.transforms)  # pragma: no cover
        elif isinstance(other, InvertibleTransform):
            return InvertibleCombinedTransform(self.transforms + [other])
        else:
            # Return a regular Combined Transform (not of invertible type)
            return CombinedTransform.extend(self, other)  # pragma: no cover

    def _backward(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a transform in reverse.

        Args:
            df, a DataFrame
        """
        for transform in self.transforms[::-1]:
            df = transform.backward(df)
        return df


class RescaleTransform(UnivariateFunctionTransform):
    """Rescales a column by another specified column."""

    def __init__(
        self, columns: Sequence[str], rescaling_columns: Sequence[str], keep_rescaling_col: bool = False
    ) -> None:
        """
        Args:
            columns, a sequence of columns to rescale (numerators)
            rescaling_columns, a sequence of columns to rescale by (denominators)
            keep_rescaling_column (bool), whether or not to retain the rescaling column in the dataframe
        """
        super().__init__(columns=columns)
        self.rescaling_columns = rescaling_columns
        self.keep_rescaling_col = keep_rescaling_col
        # Validate init args
        assert len(columns) == len(rescaling_columns)

    def _forward(self, df: pd.DataFrame) -> pd.DataFrame:  # not Tuple[pd.DataFrame, List[str]]:
        """Creates new columns with rescaled inputs in the following way:
        For each column corresponding to an input in config.inputs, divide the values in that column by the values
        in the corresponding column specified in config.rescale_input_columns.

        Args:
            config: Data Settings config
            df: DataFrame with data examples

        Returns:
            [type]: [description]
        """
        # Copy over non-transformed columns to resulting dataframe
        df_res = df.drop(list(self.columns), axis="columns", errors="ignore").copy()
        if not self.keep_rescaling_col:
            df_res = df_res.drop(list(self.rescaling_columns), axis="columns", errors="ignore")

        for col, re_col in zip(self.columns, self.rescaling_columns):
            if col in df.columns:
                new_column_name = self.new_col_name(col)
                df_res[new_column_name] = df[col].div(df[re_col].values, axis=0)  # type: ignore # auto

                logging.info(f"- Rescaling Column:  {col} -> {new_column_name}")
        return df_res  # type: ignore # auto

    def new_col_name(self, column: str) -> str:
        """Generate a new column name that incorporates the name of the rescaling column."""

        assert column in self.columns
        rescale_column = self.rescaling_columns[self.columns.index(column)]
        return f"{column} / {rescale_column}"


class LogTransform(InvertibleUnivariateFunctionTransform):
    """Transform for taking (base 10) logarithms. Allows for offsetting the data before taking the logarithm (to shift
    it into the (0, inf) range of the log operation).

    The offsets for each column will be _added_ to the values before taking the logarithm.
    """

    def __init__(self, columns: Sequence[str], offsets: Sequence[float], drop_nonpositive: bool = False) -> None:
        """
        Args:
            columns, a sequence of columns to log-transform
            offsets, a sequence of offsets
            drop_nonpositive, whether or not to drop columns that become negative after applying offsets
        """
        super().__init__(columns=columns)
        self.offsets = offsets
        self.drop_nonpositive = drop_nonpositive

    def _forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply LogTransform to supplied DataFrame

        Args:
            df, a DataFrame
        """
        # Check if column inputs within domain for log-transform (positive)
        valid_input_rows = self._inputs_within_bounds(df=df)
        # Copy over non-transformed columns to resulting dataframe and (possibly) remove invalid input rows
        df_res = df.drop(list(self.columns), axis="columns", errors="ignore").copy()[valid_input_rows]
        for i, col in enumerate(self.columns):
            if col in df.columns:
                df_res[self.new_col_name(col)] = np.log10(  # type: ignore # auto
                    df[valid_input_rows][col].values + self.offsets[i]  # type: ignore # auto
                )  # type: ignore # auto
        return df_res  # type: ignore # auto

    def _inputs_within_bounds(self, df: pd.DataFrame) -> np.ndarray:
        """
        Check if column inputs within domain for log-transform (positive)
        """
        valid_input_rows = np.ones([len(df)], dtype=bool)
        for i, col in enumerate(self.columns):
            if col not in df.columns:
                continue
            nonpositive_inputs = df[col].values + self.offsets[i] <= 0  # type: ignore # auto
            if np.any(nonpositive_inputs):  # pragma: no cover
                if self.drop_nonpositive:
                    # Specify the incompatible rows from the dataset to drop
                    valid_input_rows[nonpositive_inputs] = False
                else:
                    raise ValueError(
                        f"The inputs in column '{col}' are out of domain for a log-transform (some are non-positive). "
                        f"The min values is {df[col].values.min()} "  # type: ignore # auto
                        f"with a {self.offsets[i]} offset to be added. "
                        f"Consider adding a larger offset, or set the drop_nonpositive flag to True."
                    )
        return valid_input_rows

    def new_col_name(self, column: str) -> str:
        """Generate a new column name that incorporates a 'log'."""
        assert column in self.columns
        return f"log({column})"

    def _backward(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Invert LogTransform (exponentiate then subtract offset) from supplied DataFrame.

        Args:
            df, a DataFrame
        """
        # Copy over columns which don't need inverting to resulting dataframe
        df_res = df.drop(self.transformed_col_names(self.columns), axis="columns", errors="ignore").copy()
        for i, col in enumerate(self.columns):
            if self.new_col_name(col) in df.columns:
                df_res[col] = np.power(10, df[self.new_col_name(col)].values) - self.offsets[i]  # type: ignore # auto
        return df_res  # type: ignore # auto

    def prev_col_name(self, column: str) -> str:
        """Remove the 'log' from the column name"""
        assert column in self.transformed_col_names(self.columns)
        # Get rid of the "log(" at the start and ")" at the end
        input_col_name = column[4:-1]
        return input_col_name


class ShiftTransform(InvertibleUnivariateFunctionTransform):
    """Shifts (adds a given offset to) all values in a column.

    If constructed using the .from_df() constructor, the offset will be inferred as minus the min. of the values,
    such that the minimum of the transformed values is always 0. If combined with the MaxNormaliseTransform,
    these will normalise the range of the variables to [0, 1].
    """

    def __init__(self, columns: Sequence[str], offsets: Sequence[float]) -> None:
        """
        Args:
            columns, a sequence of columns
            offsets, a sequence of offsets
        """
        super().__init__(columns=columns)
        self.offsets = offsets
        assert len(self.columns) == len(self.offsets), "Number of offsets doesn't match the number of columns"

    @classmethod
    def from_df(cls, columns: Sequence[str], df: pd.DataFrame) -> "ShiftTransform":
        """Constructor method for creating a shift transform such that the ranges of the transformed values start at 0,
        i.e. the shift transform will subtract the minimum of the values in each column.

        Args:
            columns: Columns to shift-normalise (subtract the minimum from)
            df: DataFrame (with at least those columns specified by the columns argument) from which the offsets to
                normalise with will be inferred

        Returns:
            ShiftTransform: A ShiftTransform with offsets set to minues the minimum of the values in each column.
        """
        offsets = -df[columns].min(axis=0)  # type: ignore # auto
        assert offsets is not None
        return cls(columns=columns, offsets=offsets)

    def _forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ShiftTransform to supplied DataFrame.

        Args:
            df, a DataFrame
        """
        # Copy over non-transformed columns to resulting dataframe
        df_res = df.drop(list(self.columns), axis="columns", errors="ignore").copy()
        for i, col in enumerate(self.columns):
            if col in df.columns:
                # Divide each input by the normalising scales
                df_res[self.new_col_name(col)] = df[col].values + self.offsets[i]  # type: ignore # auto
        return df_res  # type: ignore # auto

    def new_col_name(self, column: str) -> str:
        """Generate a new column name that incorporates the applied offset."""
        assert column in self.columns
        col_idx = self.columns.index(column)
        if self.offsets[col_idx] == 0.0:
            return column  # pragma: no cover
        elif self.offsets[col_idx] > 0:
            return f"({column} + {self.offsets[col_idx]:.3g})"
        else:
            # If: self.offsets[col_idx] < 0:
            return f"({column} - {-self.offsets[col_idx]:.3g})"

    def _backward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ShiftTransform in reverse, subtracting the offsets, to supplied DataFrame.
        Args:
            df, a DataFrame
        """
        # Copy over columns which don't need inverting to resulting dataframe
        df_res = df.drop(self.transformed_col_names(self.columns), axis="columns", errors="ignore").copy()
        for i, col in enumerate(self.columns):
            if self.new_col_name(col) in df.columns:
                df_res[col] = df[self.new_col_name(col)].values - self.offsets[i]  # type: ignore # auto
        return df_res  # type: ignore # auto

    def prev_col_name(self, column: str) -> str:
        """Remove the offset from the column name"""
        assert column in self.transformed_col_names(self.columns)
        col_idx = self.transformed_col_names(self.columns).index(column)

        if self.offsets[col_idx] == 0.0:
            input_col_name = column  # pragma: no cover
        # If offset != 0, remove the "- {offset}" or "+ {offset}" part as well
        else:
            if self.offsets[col_idx] > 0:
                input_col_name = "".join(column.split(" + ")[:-1])  # pragma: no cover
            else:
                # If: self.offsets[col_idx] < 0:
                input_col_name = "".join(column.split(" - ")[:-1])
            # Remove parantheses
            input_col_name = input_col_name[1:]

        return input_col_name


class MaxNormaliseTransform(InvertibleUnivariateFunctionTransform):
    """Divides all values in a column by a scale. If constructed using the .from_df() constructor, the scale will
    be inferred as the maximum value in a column. If applied after the ShiftTransform.from_df(),
    the column will be normalised to the [0, 1] range.
    """

    def __init__(self, columns: Sequence[str], norm_scales: Sequence[float]) -> None:
        """
        Args:
            columns, a sequence of columns
            norm_scales, a sequence of normalising values
        """
        super().__init__(columns=columns)
        self.norm_scales = np.array(norm_scales)
        # Validate init args
        assert len(self.columns) == len(self.norm_scales), "Number of scales doesn't match the number of columns"
        if not np.all(self.norm_scales > 0):
            raise ValueError(  # pragma: no cover
                f"Scaling constants must be positive, but they are:\n{self.norm_scales}\n"
                f"corresponding to columns:\n{self.columns}"
            )

    @classmethod
    def from_df(cls, columns: Sequence[str], df: pd.DataFrame) -> "MaxNormaliseTransform":
        """Constructor method for creating a normalising transform where the scales to normalise with are the max.
        values for each column in a given dataframe.

        Args:
            columns: Columns to normalise (divide by the corresponding scale)
            df: DataFrame (with at least those columns specified by the columns argument) from which the scales to
                normalise with will be inferred

        Returns:
            MaxNormaliseTransform: A MaxNormaliseTransform object with scales inferred from the DataFrame
        """
        scales = df[columns].max(axis=0)
        # Check if for any of the columns to normalise, the max value is 0.
        # Raise exception if so as the column can't be normalised using a scale of 0
        some_non_positive_scales = np.any(scales <= 0)  # type: ignore # auto
        if some_non_positive_scales:  # pragma: no cover
            # Construct the error message
            offending_columns = [col for col, scale in zip(columns, scales) if scale <= 0]  # type: ignore # auto
            error_msg = (
                f"Cannot normalise when the maximum value of the inputs is less than or equal to 0, "
                f"which is the case for columns:\n{offending_columns}"
            )
            all_inputs_same = (
                df[offending_columns].max(axis=0).values  # type: ignore # auto
                - df[offending_columns].min(axis=0).values  # type: ignore # auto
            ) == 0  # type: ignore # auto
            if np.any(all_inputs_same):
                # The range of the input in some columns is zero.
                error_msg += (
                    f"\nFor some columns, all inputs are the same, which could lead to an error if "
                    f"normalising to [0, 1] range with ShiftTransform and MaxNormaliseTransform. These columns are:"
                    f"{[col for col, range_same in zip(offending_columns, all_inputs_same) if range_same]}"
                )
            raise ValueError(error_msg)
        return cls(columns=columns, norm_scales=scales)  # type: ignore # scales

    def _forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the MaxNormalise transform to supplied DataFrame.

        Args:
            df, a DataFrame
        """
        # Copy over non-transformed columns to resulting dataframe
        df_res = df.drop(list(self.columns), axis="columns", errors="ignore").copy()
        for i, col in enumerate(self.columns):
            if col in df.columns:
                # Divide each input by the normalising scales
                df_res[self.new_col_name(col)] = df[col].values / self.norm_scales[i]
        return df_res  # type: ignore # auto

    def new_col_name(self, column: str) -> str:
        """Generate a new column name that incorporates the normalising constant."""
        assert column in self.columns
        col_idx = self.columns.index(column)
        return f"{column} / {self.norm_scales[col_idx]:.3g}"

    def _backward(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply MaxNormaliseTransform in reverse, multiplying by the normalising constant, to supplied DataFrame.

        Args:
            df, a DataFrame
        """
        # Copy over columns which don't need inverting to resulting dataframe
        df_res = df.drop(self.transformed_col_names(self.columns), axis="columns", errors="ignore").copy()
        for i, col in enumerate(self.columns):
            if self.new_col_name(col) in df.columns:
                df_res[col] = df[self.new_col_name(col)].values * self.norm_scales[i]
        return df_res  # type: ignore # auto

    def prev_col_name(self, column: str) -> str:
        """Remove the normalising constant from the column name"""
        assert column in self.transformed_col_names(self.columns)
        return "".join(column.split(" / ")[:-1])


class OneHotTransform(InvertibleTransform):
    """This class takes a dataframe with some categorical input columns, e.g. ['colour', 'race'],
    and converts them to a dataframe with a onehot encoding of these variables, e.g. one with columns
    ['colour: red', 'colour: blue', 'race: terrier', 'race: german shepherd'], with values in {0, 1}.
    """

    def __init__(self, columns: Sequence[str], categorical_space: "OrderedDict[str, List]") -> None:
        """
        Args:
            columns, a sequence of column names
            categorical_space, an ordered dictionary of categorical variables with their categories
        """
        super().__init__(columns)
        self.categorical_space = categorical_space
        # Assert categorical space matches columns (both in order and values)
        for col, space_col in zip(columns, categorical_space.keys()):
            assert col == space_col

    @classmethod
    def from_df(
        cls, columns: Sequence[str], df: pd.DataFrame, partial_categ_space: Optional[Dict[str, List]] = None
    ) -> "OneHotTransform":
        """
        Infer categorical space (the possible categories) from the DataFrame given for all the categorical inputs
        not specified in 'partial_categ_space'.
        """
        if partial_categ_space is None:
            partial_categ_space = {}
        categorical_space = OrderedDict()
        for column in columns:
            inferred_categories = list(df[column].unique())
            if column in partial_categ_space:  # pragma: no cover
                assert set(inferred_categories).issubset(set(partial_categ_space[column]))
                categorical_space[column] = partial_categ_space[column]
            else:
                categorical_space[column] = inferred_categories
        return cls(columns, categorical_space)

    def _forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot transform to supplied DataFrame.

        Args:
            df, a DataFrame
        """
        # Copy over non-transformed columns to resulting dataframe
        df_res = df.drop(list(self.columns), axis="columns", errors="ignore").copy()
        for col in self.columns:
            if col in df.columns:
                for category in self.categorical_space[col]:
                    # Add a new column 'column:category' with 1s where values in column == category, 0s otherwise
                    df_res[self.new_col_name(col, category)] = (df[col] == category).values.astype(float)
        return df_res  # type: ignore # auto

    def new_col_name(self, column: str, category: str) -> str:
        """Generate a new column name that includes the category."""
        assert column in self.columns
        assert category in self.categorical_space[column]
        return f"{column}:{category}"

    def _backward(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        """
        Apply OneHotTransform in reverse, mapping one-hot columns back to a categorical columns, to supplied DataFrame.

        Args:
            df, a DataFrame
        """
        # Copy over columns which don't need inverting to resulting dataframe
        df_res = df.drop(self.transformed_col_names(self.columns), axis="columns", errors="ignore").copy()
        for col in self.columns:
            # Check if onehot encoding for this column in dataframe
            if self.new_col_name(col, self.categorical_space[col][0]) in df.columns:
                # Create the column and fill with NaNs, then replace them with category values by iterating over categs
                df_res[col] = np.nan
                for category in self.categorical_space[col]:
                    if self.new_col_name(col, category) not in df.columns:
                        raise ValueError(f"Category: {category} missing from onehot encoding")  # pragma: no cover
                    # Where the field for this category in onehot encoding is 1, set column value to category
                    ones = df[self.new_col_name(col, category)].values
                    df_res.loc[ones.astype(bool), col] = category  # type: ignore
        return df_res  # type: ignore # auto

    def transformed_col_names(self, columns: Sequence[str]) -> List[str]:
        """Generate column names for one-hot encodings that incorporate the category names."""
        transformed_cols = [col for col in columns if col not in self.columns]
        for col in columns:
            if col in self.columns:
                for category in self.categorical_space[col]:
                    transformed_cols.append(self.new_col_name(col, category))
        return transformed_cols

    @property
    def onehot_column_names(self) -> List[str]:
        """Returns a list of column names for the one-hot encoded categorical variables. I.e. if the category columns
        have names ['Col1', 'Col2'...] and each of the inputs in each columns takes on values in some categorical space,
        then the resulting columns names will be:
            ['Col1: label11', 'Col1: label12', ..., 'Col2: label21', ...]
        where 'labelXX' are the values that inputs in that column can take.
        """
        return self.transformed_col_names(self.columns)
